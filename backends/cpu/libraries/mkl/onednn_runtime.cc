#include "onednn_runtime.h"
#include "core/var.h"
#include <dnnl.hpp>
#include <dnnl_version.h>
#include <list>
#include <unordered_map>
#include <algorithm>
#include <cstring>

#if DNNL_VERSION_MAJOR < 3
#error "Jittor's oneDNN backend requires oneDNN v3 or newer"
#endif

namespace jittor {
namespace {
using namespace dnnl;
using Dims = memory::dims;
using Args = std::unordered_map<int, memory>;
struct Key {
    int kind = 0;
    std::array<int64, 24> dimensions{};
    std::array<string, 3> formats;
    bool operator==(const Key& rhs) const {
        return kind == rhs.kind && dimensions == rhs.dimensions && formats == rhs.formats;
    }
};
struct Step { primitive operation; Args arguments; };
struct Plan {
    primitive_desc descriptor;
    convolution_forward::primitive_desc forward_hint;
    std::array<memory, 3> user;
    vector<Step> steps;
    size_t scratch_bytes = 0;
};
struct Runtime {
    engine eng{engine::kind::cpu, 0};
    stream execution{eng};
    std::list<std::pair<Key, std::unique_ptr<Plan>>> plans;
    uint64 builds = 0, hits = 0, executions = 0;
    size_t scratch_bytes = 0;
};
Runtime& runtime() { static thread_local Runtime state; return state; }
constexpr size_t max_plans = 32;
constexpr size_t max_scratch_bytes = 64 * 1024 * 1024;

memory::desc user_desc(const Dims& shape, const string& order, const string& canonical) {
    Dims strides(shape.size());
    int64 running = 1;
    for (int i=int(order.size())-1; i>=0; --i) {
        size_t axis = canonical.find(order[i]);
        USER_CHECK(axis < shape.size()) << "oneDNN invalid layout" << order;
        strides[axis] = running;
        running *= shape[axis];
    }
    return {shape, memory::data_type::f32, strides};
}

memory input(Plan& plan, int index, const memory::desc& selected, engine& eng) {
    auto buffer = plan.user[index];
    if (buffer.get_desc() == selected) return buffer;
    memory internal(selected, eng);
    plan.scratch_bytes += selected.get_size();
    plan.steps.push_back({reorder(buffer, internal), {{DNNL_ARG_FROM, buffer}, {DNNL_ARG_TO, internal}}});
    return internal;
}

void finish(Plan& plan, primitive operation, Args arguments, int index,
            int argument, const memory::desc& selected, engine& eng) {
    auto target = plan.user[index];
    if (target.get_desc() == selected) {
        arguments.emplace(argument, target);
        plan.steps.push_back({operation, move(arguments)});
    } else {
        memory internal(selected, eng);
        plan.scratch_bytes += selected.get_size();
        arguments.emplace(argument, internal);
        plan.steps.push_back({operation, move(arguments)});
        plan.steps.push_back({reorder(internal, target), {{DNNL_ARG_FROM, internal}, {DNNL_ARG_TO, target}}});
    }
}

std::unique_ptr<Plan> make_conv(const OneDnnConvSpec& spec, Runtime& state) {
    auto plan = std::unique_ptr<Plan>(new Plan);
    auto& eng = state.eng;
    Dims src(spec.source.begin(), spec.source.end());
    Dims weight(spec.weights.begin(), spec.weights.end());
    Dims dst(spec.destination.begin(), spec.destination.end());
    if (spec.groups > 1) weight = {spec.groups, weight[0]/spec.groups, weight[1], weight[2], weight[3]};
    Dims stride(spec.stride.begin(), spec.stride.end());
    Dims pad(spec.padding.begin(), spec.padding.end());
    Dims dilation{spec.dilation[0]-1, spec.dilation[1]-1};
    auto xmd = user_desc(src, spec.xformat, "abcd");
    auto wmd = user_desc(weight, spec.groups > 1 ? "goihw" : spec.wformat,
                        spec.groups > 1 ? "goihw" : "oihw");
    auto ymd = user_desc(dst, spec.yformat, "abcd");
    plan->user = {memory(xmd, eng, DNNL_MEMORY_NONE), memory(wmd, eng, DNNL_MEMORY_NONE),
                  memory(ymd, eng, DNNL_MEMORY_NONE)};
    auto xs = memory::desc(src, memory::data_type::f32, memory::format_tag::any);
    auto ws = memory::desc(weight, memory::data_type::f32, memory::format_tag::any);
    auto ys = memory::desc(dst, memory::data_type::f32, memory::format_tag::any);
    // The same training hint/layout contract serves all three directions.
    auto forward = convolution_forward::primitive_desc(eng, prop_kind::forward_training,
        algorithm::convolution_auto, xs, ws, ys, stride, dilation, pad, pad);
    plan->forward_hint = forward;
    if (spec.kind == 0) {
        plan->descriptor = forward;
        auto x = input(*plan, 0, forward.src_desc(), eng);
        auto w = input(*plan, 1, forward.weights_desc(), eng);
        finish(*plan, convolution_forward(forward), {{DNNL_ARG_SRC, x}, {DNNL_ARG_WEIGHTS, w}},
               2, DNNL_ARG_DST, forward.dst_desc(), eng);
    } else if (spec.kind == 1) {
        auto pd = convolution_backward_data::primitive_desc(eng, algorithm::convolution_direct,
            xs, ws, ys, stride, dilation, pad, pad, forward);
        plan->descriptor = pd;
        auto w = input(*plan, 1, pd.weights_desc(), eng);
        auto dy = input(*plan, 2, pd.diff_dst_desc(), eng);
        finish(*plan, convolution_backward_data(pd), {{DNNL_ARG_WEIGHTS, w}, {DNNL_ARG_DIFF_DST, dy}},
               0, DNNL_ARG_DIFF_SRC, pd.diff_src_desc(), eng);
    } else {
        auto pd = convolution_backward_weights::primitive_desc(eng, algorithm::convolution_direct,
            xs, ws, ys, stride, dilation, pad, pad, forward);
        plan->descriptor = pd;
        auto x = input(*plan, 0, pd.src_desc(), eng);
        auto dy = input(*plan, 2, pd.diff_dst_desc(), eng);
        finish(*plan, convolution_backward_weights(pd), {{DNNL_ARG_SRC, x}, {DNNL_ARG_DIFF_DST, dy}},
               1, DNNL_ARG_DIFF_WEIGHTS, pd.diff_weights_desc(), eng);
    }
    return plan;
}

std::unique_ptr<Plan> make_matmul(const Key& key, Runtime& state) {
    auto plan = std::unique_ptr<Plan>(new Plan);
    auto& d = key.dimensions;
    int64 batch=d[0], n=d[1], m=d[2], k=d[3];
    bool ta=d[4], tb=d[5];
    memory::desc a({batch,n,m}, memory::data_type::f32, Dims{n*m, ta ? 1 : m, ta ? n : 1});
    memory::desc b({batch,m,k}, memory::data_type::f32, Dims{m*k, tb ? 1 : k, tb ? m : 1});
    memory::desc c({batch,n,k}, memory::data_type::f32, Dims{n*k,k,1});
    auto pd = matmul::primitive_desc(state.eng, a, b, c);
    plan->descriptor = pd;
    plan->user = {memory(a, state.eng, DNNL_MEMORY_NONE), memory(b, state.eng, DNNL_MEMORY_NONE),
                  memory(c, state.eng, DNNL_MEMORY_NONE)};
    plan->steps.push_back({matmul(pd), {{DNNL_ARG_SRC, plan->user[0]},
        {DNNL_ARG_WEIGHTS, plan->user[1]}, {DNNL_ARG_DST, plan->user[2]}}});
    return plan;
}

template<class Build>
void execute(const Key& key, void* a, void* b, void* c, Build build) {
    auto& state = runtime();
    auto found = std::find_if(state.plans.begin(), state.plans.end(),
                             [&](const auto& item) { return item.first == key; });
    if (found == state.plans.end()) {
        auto plan = build(state);
        state.scratch_bytes += plan->scratch_bytes;
        state.plans.emplace_front(key, move(plan));
        ++state.builds;
    } else {
        state.plans.splice(state.plans.begin(), state.plans, found);
        ++state.hits;
    }
    auto& plan = *state.plans.front().second;
    struct Unbind {
        Plan& plan;
        ~Unbind() {
            for (auto& memory : plan.user) {
                auto status = dnnl_memory_set_data_handle(memory.get(), DNNL_MEMORY_NONE);
                if (status != dnnl_success) LOGe << "oneDNN data-handle cleanup failed:" << int(status);
            }
        }
    } unbind{plan};
    plan.user[0].set_data_handle(a);
    plan.user[1].set_data_handle(b);
    plan.user[2].set_data_handle(c);
    for (auto& step : plan.steps) step.operation.execute(state.execution, step.arguments);
    state.execution.wait();
    ++state.executions;
    // Keep the current plan alive through Unbind; large one-off plans are
    // evicted on the next call, with at most one oversized live plan.
    while (state.plans.size() > 1 && (state.plans.size() > max_plans || state.scratch_bytes > max_scratch_bytes)) {
        state.scratch_bytes -= state.plans.back().second->scratch_bytes;
        state.plans.pop_back();
    }
}
}

void check_onednn_conv_args(Var* a, Var* b, int sh, int sw, int ph, int pw,
    int dh, int dw, int groups, const string& xf, const string& wf, const string& yf) {
    USER_CHECK(a->dtype() == ns_float32 && b->dtype() == ns_float32)
        << "oneDNN convolution supports float32 inputs only";
    USER_CHECK(a->shape.size() == 4 && b->shape.size() == 4)
        << "oneDNN convolution requires rank-4 inputs";
    USER_CHECK(sh>0 && sw>0 && dh>0 && dw>0 && ph>=0 && pw>=0 && groups>0)
        << "oneDNN convolution requires positive stride/dilation/groups and non-negative padding";
    auto valid = [](string value, string canonical) {
        std::sort(value.begin(), value.end()); std::sort(canonical.begin(), canonical.end());
        return value == canonical;
    };
    USER_CHECK(valid(xf,"abcd") && valid(wf,"oihw") && valid(yf,"abcd"))
        << "oneDNN convolution invalid layout" << xf << wf << yf;
    USER_CHECK(groups == 1 || wf == "oihw") << "oneDNN grouped convolution requires oihw weights";
}
OneDnnConvSpec onednn_conv_spec(int kind, Var* x, Var* w, Var* y,
    int sh, int sw, int ph, int pw, int dh, int dw, int groups,
    const string& xf, const string& wf, const string& yf) {
    OneDnnConvSpec spec;
    spec.kind=kind; spec.stride={sh,sw}; spec.padding={ph,pw}; spec.dilation={dh,dw};
    spec.groups=groups; spec.xformat=xf; spec.wformat=wf; spec.yformat=yf;
    for (int i=0; i<4; ++i) {
        spec.source[i]=x->shape[xf.find("abcd"[i])];
        spec.weights[i]=w->shape[wf.find("oihw"[i])];
        spec.destination[i]=y->shape[yf.find("abcd"[i])];
    }
    return spec;
}
void onednn_conv_execute(const OneDnnConvSpec& spec, void* x, void* w, void* y) {
    Key key;
    key.kind = spec.kind;
    size_t i=0;
    for (auto v : spec.source) key.dimensions[i++]=v;
    for (auto v : spec.weights) key.dimensions[i++]=v;
    for (auto v : spec.destination) key.dimensions[i++]=v;
    for (auto v : spec.stride) key.dimensions[i++]=v;
    for (auto v : spec.padding) key.dimensions[i++]=v;
    for (auto v : spec.dilation) key.dimensions[i++]=v;
    key.dimensions[i]=spec.groups;
    key.formats = {spec.xformat, spec.wformat, spec.yformat};
    execute(key, x, w, y, [&](Runtime& state) { return make_conv(spec, state); });
}
void onednn_matmul_execute(int64 batch, int64 n, int64 m, int64 k,
                           bool ta, bool tb, void* a, void* b, void* c) {
    if (!batch || !n || !k) return;
    if (!m) { std::memset(c, 0, size_t(batch*n*k)*sizeof(float)); return; }
    Key key;
    key.kind = 3;
    key.dimensions = {batch,n,m,k,int64(ta),int64(tb)};
    execute(key, a, b, c, [&](Runtime& state) { return make_matmul(key, state); });
}
vector<int64> onednn_cache_info() {
    auto& state=runtime();
    return {int64(state.builds), int64(state.hits), int64(state.executions),
            int64(state.plans.size()), int64(state.scratch_bytes)};
}
void onednn_cache_clear() { auto& state=runtime(); state.plans.clear(); state.scratch_bytes=0; }
vector<int> onednn_version() {
    const auto* version=dnnl_version();
    return {version->major,version->minor,version->patch};
}
}
