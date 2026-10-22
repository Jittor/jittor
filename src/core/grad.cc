// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "bindings/pybind/py_var_tracer.h"
#include "core/grad.h"
#include "runtime/launch_diagnostics.h"
#include "core/var.h"
#include "core/op.h"
#include "core/graph.h"
#include "runtime/node_index.h"
#include "ops/op_register.h"
#include "core/var_holder.h"
#include "runtime/submission_pipeline.h"
#include <atomic>
#include <algorithm>

namespace jittor {

#define PREVENT_LARGE_FUSED_OP 16

namespace {
struct StepBackwardProbe {
    uint64 token;
    unordered_set<int64> variables;
    explicit StepBackwardProbe(uint64 token) : token(token) {}
};
thread_local vector<StepBackwardProbe> step_backward_probes;
std::atomic<uint64> next_step_backward_probe{0};
}

uint64 begin_step_backward_probe() {
    uint64 token = next_step_backward_probe.fetch_add(1, std::memory_order_relaxed) + 1;
    step_backward_probes.emplace_back(token);
    return token;
}

vector<int64> end_step_backward_probe(uint64 token) {
    USER_CHECK(!step_backward_probes.empty() && step_backward_probes.back().token == token)
        << "step backward probes must end on their creating thread in LIFO order";
    auto variables = move(step_backward_probes.back().variables);
    step_backward_probes.pop_back();
    vector<int64> result(variables.begin(), variables.end());
    std::sort(result.begin(), result.end());
    return result;
}

AutogradPolicyState autograd_policy;

void set_autograd_policy(
    bool stop_outputs_when_inputs_stopped,
    bool preserve_requires_grad_on_assignment
) {
    autograd_policy.stop_outputs_when_inputs_stopped =
        stop_outputs_when_inputs_stopped;
    autograd_policy.preserve_requires_grad_on_assignment =
        preserve_requires_grad_on_assignment;
}

int get_autograd_policy() {
    return int(autograd_policy.stop_outputs_when_inputs_stopped)
        | (int(autograd_policy.preserve_requires_grad_on_assignment) << 1);
}

DECLARE_FLAG(int, auto_mixed_precision_level);

static auto make_binary = op_constructor<VarPtr, Var*, Var*, NanoString>("binary");
static auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
static auto make_number = op_constructor<VarPtr, float, Var*>("number");
static auto make_reduce = op_constructor<VarPtr, Var*, NanoString, uint, uint>("reduce");

static VarPtr normalize_recorded_gradient(Var* target, VarPtr value) {
    if (!value || !target->recorded_gradient_shape) return value;
    const auto& expected = *target->recorded_gradient_shape;
    const auto& actual = value->shape;
    USER_CHECK(actual.size() >= expected.size())
        << "invalid gradient shape" << actual << "; expected shape compatible with" << expected;
    uint reduce = 0, keep = 0;
    uint leading = actual.size() - expected.size();
    for (uint axis=0; axis<actual.size(); ++axis) {
        if (axis < leading) reduce |= 1u << axis;
        else {
            auto dimension = expected[axis-leading];
            USER_CHECK(actual[axis] == dimension || dimension == 1)
                << "invalid gradient shape" << actual << "; expected shape compatible with" << expected;
            if (actual[axis] != dimension) {
                reduce |= 1u << axis;
                keep |= 1u << axis;
            }
        }
    }
    return reduce ? make_reduce(value.ptr, ns_add, reduce, keep) : move(value);
}

#ifdef _WIN32
template<class T> struct StackIniter {
    T* a;
    int n;
    inline StackIniter(T* a, int n) :a(a), n(n) {
        for (int i=0; i<n; i++)
            new(a+i) T();
    }
    inline ~StackIniter() {
        for (int i=0; i<n; i++)
            a[i].~T();
    }
};

#define STACK_ALLOC2(T, a, n) T* a = (T*)_alloca(sizeof(T)*(n)); StackIniter<T> __init_##a(a, n);
#else
#define STACK_ALLOC2(T, a, n) T a[n]
#endif

struct AmpGradGuard {
    int amp_reg_bk;
    AmpGradGuard(Op* op) {
        amp_reg_bk = amp_reg;
        // Mirror only the seven amp bits the constructor wrote (op.cc). Reading
        // the field by width rather than shifting the raw word is what keeps
        // _custom_flag and the requires_grad bookkeeping bits out of amp_reg;
        // the two sides now name the same field instead of the same number.
        amp_reg |= op->flag(OpFlags::_prefer_32, OpFlags::_amp_nbits);
    }

    ~AmpGradGuard() {
        amp_reg = amp_reg_bk;
    }
};

VarPtr make_grad(Op* op, Var* out, Var* dout, Var* x, int x_index) {
    if (dout == nullptr) return nullptr;
    if (x_index<0) return nullptr;
    LOGvvvv << "Make grad op:" >> op->name() << "inputs:" >> op->inputs()
        << "out:" >> out << "dout:" >> dout << "x:" >> x << "xid:" >> x_index;
    AmpGradGuard agg(op);
    Float32PrecisionScope precision_scope(op->float32_precision);
    LaunchOriginScope origin_scope(op->launch_origin);
    auto dx = op->grad(out, dout, x, x_index);
    // A null dx is an ordinary path, not an error: floor/round/ceil, mod,
    // floor_divide, the bitwise ops and the default Op::grad all return one.
    // The guard used to test the input x and then dereference the result.
    if (dx && x->loop_options)
        dx->loop_options = x->loop_options;
    return dx;
}

// Whether autograd is willing to send a gradient into `v` at all. The two
// flags are the pair VarHolder::get_requires_grad reads and the pair
// grad()'s bfs_backward stops on, which is why they are one predicate here.
inline static bool requires_grad(Var* v) {
    return !v->is_stop_grad()
        && !v->flag(VarFlags::_requires_grad_disabled);
}

Op* backward_grad_fn(Var* v) {
    ASSERT(v != nullptr);
    if (!requires_grad(v)) return nullptr;
    Op* op = v->input();
    // No producer: a graph source, or a producer that has already been freed --
    // and it can only have been freed if nothing needed it for backward, which
    // is the same answer.
    if (!op) return nullptr;
    // detach() stops the *op*, not the var it produced (ops/clone_op.cc), so a
    // detached var requires grad and is still a leaf. Reading the var's own
    // flag would have missed this and called it a non-leaf.
    if (op->is_stop_grad()) return nullptr;
    for (auto& e : op->_inputs) {
        // Control dependency: make_grad refuses a negative index, so this edge
        // exists for ordering only.
        if (e.reverse().index < 0) continue;
        Var* in = e.node->var();
        // A later reversible freeze of an input does not erase the producer
        // already recorded for this output. Edges disabled at construction
        // are checked below; permanent stop-grad still ends the graph.
        if (in->is_stop_grad()) continue;
        // An unrecorded view can require grad and create a producer even
        // though this particular input has no backward edge. This differs
        // from an input explicitly frozen when the op was constructed.
        if (is_requires_grad_disabled_edge(in, op)
                && !lookup_unrecorded_view_edge(in, op)) continue;
        return op;
    }
    // A recorded view can rebase over a store made while its source was
    // frozen. It still has a producer and accepts its own gradient, although
    // the rebased producer's input-edge snapshot deliberately carries no
    // gradient back to that source, even after the source is enabled again.
    if (v->autograd_alias && v->autograd_alias->recorded_view_history
            && !v->autograd_alias->created_without_grad
            && !v->autograd_alias->source_edge_enabled
            && v->autograd_alias->creation_version == *v->autograd_alias->version) {
        // Reversible edge snapshots may be disabled, but a permanent stop on
        // every real input still terminates this producer's native history.
        for (auto& e : op->_inputs)
            if (e.reverse().index >= 0 && !e.node->var()->is_stop_grad())
                return op;
    }
    return nullptr;
}

inline static void assign_attrs(Var* a, Var* b) {
    if (b->flag(VarFlags::_stop_fuse))
        a->set_flag(VarFlags::_stop_fuse);
}

static void apply_tensor_hook(Var* var, VarPtr& gradient) {
    // Keep the callback alive if its Python hook removes itself or rebinds
    // the parameter while running. A missing gradient never fires a hook.
    auto callback = var->grad_hook;
    if (!callback || !gradient) return;
    if (gradient->dtype() != var->dtype())
        gradient = make_unary(gradient, var->dtype());
    Var* incoming = gradient.ptr;
    VarPtr replacement;
    callback->func(1, &incoming, 1, &replacement);
    if (!replacement) return;
    USER_CHECK(replacement->shape == gradient->shape)
        << "Tensor gradient hook changed the gradient shape";
    USER_CHECK(replacement->dtype() == gradient->dtype())
        << "Tensor gradient hook changed the gradient dtype";
    USER_CHECK(replacement->device_id == gradient->device_id)
        << "Tensor gradient hook changed the gradient device";
    gradient = move(replacement);
}

DEFINE_FLAG(int, missing_grad_error, 0, "Raise instead of warning when a target of grad receives no gradient at all and is filled with zeros.");

void warn_grad_break(int i, Var* v) {
    // This used to deduplicate through a process-global map keyed on
    // v->name.c_str(). Almost every var's name is the empty string, so exactly
    // one missing gradient was ever announced per process and every later one
    // -- any var, any later call -- was silent, and training converged to the
    // wrong thing with a clean log. There is no deduplication now: one report
    // per missing target per grad() call.
    std::stringstream target;
    target << "jt.grad: target " << i << " (" << v->dtype() << v->shape;
    if (v->name.size()) target << ", named '" << v->name << "'";
    target << ") does not depend on the loss through a differentiable path";
    if (missing_grad_error) {
        USER_ERROR << target.str() >> ". Drop it from the target list, or clear"
            << "jt.flags.missing_grad_error to get a zero gradient and a"
            << "warning instead.";
    }
    user_warning(target.str() + "; its gradient is zero. Set "
                 "jt.flags.missing_grad_error=1 to raise instead.");
}

vector<VarPtr> grad(
    Var* loss,
    vector<Var*> targets,
    bool retain_graph,
    bool materialize_grads
) {
    // No auto-flush while the backward graph is half built; see
    // `SubmissionPipeline::grad_construction_depth`.
    GradConstructionScope construction(runtime_submission_pipeline());
    LOGvv << "loss:" >> loss << "targets:" >> targets;
    USER_CHECK(loss->is_float()) << "Loss should be float";
    USER_CHECK(!loss->flag(VarFlags::_first_order_only))
        << "Higher-order gradients are not supported because this loss "
        << "depends on a first-order-only gradient result.";
    for (Var* var : targets)
        USER_CHECK(var->is_float() || var->dtype().is_complex())
            << "Targets of grad should be float or complex";
    // A var whose differentiability an earlier backward gave up (see the
    // retain_graph branch at the end of this function). Reaching one means the
    // caller is backwarding through a graph that was already released.
    Var* released = loss->flag(VarFlags::_graph_freed) ? loss : nullptr;
    // successors of targets
    vector<Node*> ts(targets.begin(), targets.end());
    // bfs visit find all successors of targets
    LOGvv << "Size of successors:" << ts.size();
    bfs_forward(ts, [](Node*){ return true; });
    NodeIndex successors;
    successors.reset(ts.size());
    for (Node* node : ts) successors[node] = 1;
    vector<Node*> gnodes;
    gnodes.reserve(ts.size());
    if (successors.has(loss))
        gnodes.push_back(loss);
    bfs_backward(gnodes, [&](Node* node) {
        if (!successors.has(node))
            return false;
        if (node->is_var() && node->var()->flag(VarFlags::_graph_freed))
            released = node->var();
        if (node->is_stop_grad()
            || (node->is_var()
                && node->var()->flag(VarFlags::_requires_grad_disabled)))
            return false;
        return true;
    });
    if (released)
        USER_ERROR << "Trying to backward through the graph a second time. This"
            << "backward graph was released by an earlier backward with"
            << "retain_graph=False and cannot be walked again; continuing would"
            << "silently produce zero gradients. Pass retain_graph=True to the"
            << "first backward if you need to backward through the same graph"
            << "twice, or rebuild the forward pass. Released var:" << released;
    LOGvv << "Size of grad nodes:" << gnodes.size();
    
    vector<Node*> sorted;
    toplogical_sort_backward(gnodes, sorted, [](Node*){});
    NodeIndex grad_nodes;
    grad_nodes.reset(sorted.size());
    for (Node* node : sorted) grad_nodes[node] = 1;
    vector<Var*> gvars;
    gvars.reserve(sorted.size());
    // Position of each gradient var in `gvars`. This used to be written into
    // Node::custom_data, the per-node int the executor keeps *its* op and var
    // indices in -- and building the backward ops below re-enters run_sync
    // (Op::init does, for a vary-shape op), so the two numberings were live at
    // the same time on the same nodes.
    NodeIndex gvar_index;
    gvar_index.reset(sorted.size());
    for (Node* node : sorted)
        if (node->is_var()) {
            Var* v = node->var();
            gvar_index[v] = gvars.size();
            gvars.push_back(v);
        }
    LOGvv << "Size of grad vars:" << gvars.size();
    
    vector<VarPtr> grads(gvars.size());
    vector<VarPtr> results(targets.size());
    vector<int> target_id(targets.size());
    vector<bool> is_target(gvars.size(), false);
    unordered_map<Var*, VarPtr> observed_target_grads;
    for (int i=0; i<targets.size(); i++) {
        Var* var = targets[i];
        // Traversal follows the input-edge snapshot taken by each consumer.
        // Disabling an alias's own accumulator after an old consumer was
        // recorded must not erase that consumer's existing edge. A direct
        // unrecorded backward root still has no accumulator to seed.
        target_id[i] = (var != loss || !var->flag(VarFlags::_unrecorded_view))
            && gvar_index.has(var) ?
            gvar_index.get(var) : -1;
        if (target_id[i] >= 0) is_target[target_id[i]] = true;
    }

    if (grads.size()) {
        grads[0] = make_number(1.f, loss);
        assign_attrs(grads[0].ptr, loss);
        auto producer = backward_grad_fn(loss);
        if (loss->grad_hook && is_target[0] && producer && grad_nodes.has(producer))
            observed_target_grads[loss] = grads[0].ptr;
        if (!loss->flag(VarFlags::_unrecorded_view))
            apply_tensor_hook(loss, grads[0]);
    }

    NodeIndex consumed_grouped_ops;
    consumed_grouped_ops.reset(sorted.size());

    struct GradOpSnapshot {
        Op* op;
        int input_index;
        vector<pair<Var*, int>> outputs;
        vector<int> input_grad_ids;
    };

    // Snapshot one var's outgoing structure immediately before consuming it.
    // Grad construction may append new consumers to this same list, or add
    // outputs to an upstream op that has not been visited yet.
    for (int i=1; i<gvars.size(); i++) {
        Var* var = gvars[i];
        vector<GradOpSnapshot> outgoing;
        outgoing.reserve(var->outputs().size());
        for (auto it : var->outputs_with_index()) {
            Op* op = it.op;
            if (!grad_nodes.has(op)) continue;
            if (is_requires_grad_disabled_edge(var, op)) continue;
            if (op->flag(OpFlags::_grads) && consumed_grouped_ops.has(op))
                continue;

            GradOpSnapshot snapshot{op, it.index, {}, {}};
            snapshot.outputs.reserve(op->outputs().size());
            for (Var* out : op->outputs())
                snapshot.outputs.emplace_back(
                    out, gvar_index.has(out) ? gvar_index.get(out) : -1);

            if (op->flag(OpFlags::_grads)) {
                consumed_grouped_ops[op] = 1;
                snapshot.input_grad_ids.reserve(op->inputs().size());
                for (Var* in : op->inputs())
                    snapshot.input_grad_ids.push_back(
                        !is_requires_grad_disabled_edge(in, op)
                            && gvar_index.has(in) ? gvar_index.get(in) : -1);
            }
            outgoing.push_back(move(snapshot));
        }

        auto& grad = grads[i];
        #ifdef PREVENT_LARGE_FUSED_OP
        int gsum = 0;
        #endif
        for (const auto& snapshot : outgoing) {
            Op* op = snapshot.op;
            int n_o = snapshot.outputs.size();

            if (op->flag(OpFlags::_grads)) {
                int n_i = snapshot.input_grad_ids.size();
                STACK_ALLOC(Var*, douts, n_o);
                STACK_ALLOC2(VarPtr, dins, n_i);
                for (int i=0; i<n_o; i++) {
                    int id = snapshot.outputs[i].second;
                    if (id>=0) {
                        douts[i] = grads[id];
                    } else
                        douts[i] = nullptr;
                }
                trace_grad_op = op;
                {
                    AmpGradGuard agg(op);
                    Float32PrecisionScope precision_scope(op->float32_precision);
                    LaunchOriginScope origin_scope(op->launch_origin);
                    op->grads(douts, dins);
                }
                for (int i=0; i<n_i; i++) {
                    int id = snapshot.input_grad_ids[i];
                    if (id>=0) {
                        auto din = normalize_recorded_gradient(gvars[id], move(dins[i]));
                        auto& grad = grads[id];
                        if (din && grad) {
                            grad = make_binary(grad, din, ns_add);
                        } else
                            grad = move(din);
                    }
                }
            } else {
                for (const auto& output : snapshot.outputs) {
                    Var* out = output.first;
                    int id = output.second;
                    if (id<0) continue;
                    Var* dout = grads[id];
                    trace_grad_op = op;
                    VarPtr dvar = make_grad(
                        op, out, dout, var, snapshot.input_index);
                    dvar = normalize_recorded_gradient(var, move(dvar));
                    if (dvar && dvar->num>=0 && var->num>0 && !var->recorded_gradient_shape)
                        // var->num == 0 represents a any match var
                        ASSERT(dvar->num==var->num && dvar->shape.size()==var->shape.size())
                        << "dvar" << dvar << "var" << var;
                    if (!grad)
                        grad = move(dvar);
                    else if (dvar) {
                        grad = make_binary(grad, dvar, ns_add);
                        #ifdef PREVENT_LARGE_FUSED_OP
                        gsum ++;
                        if (gsum>=PREVENT_LARGE_FUSED_OP) {
                            // TODO: this is a dirty fix for
                            // stopping fuse lots of op together,
                            // try to find a better solution
                            grad->set_flag(VarFlags::_stop_fuse);
                        }
                        #endif
                        assign_attrs(grad.ptr, var);
                    }
                }
            }
        }
        // Same story: this var may have collected no gradient at all, in which
        // case grad is null and there is nothing to cast.
        if (grad && auto_mixed_precision_level == 3 && grad->ns != var->ns) {
            grad = make_unary(grad, var->ns);
        }
        // A non-leaf input's returned gradient is observed before its node
        // executes (and before that node's Tensor hooks), if this traversal
        // also differentiates upstream of it. At a terminal target the hooks
        // still run, and the returned gradient includes their replacement.
        if (grad && var->grad_hook && is_target[i]) {
            auto producer = backward_grad_fn(var);
            if (producer && grad_nodes.has(producer))
                observed_target_grads[var] = grad.ptr;
        }
        apply_tensor_hook(var, grad);
    }
    trace_grad_op = nullptr;
    // set zero grad
    for (size_t i=0; i<results.size(); i++) {
        Var* var = targets[i];
        VarPtr& grad = results[i];
        auto id = target_id[i];
        if (id>=0) {
            auto observed = observed_target_grads.find(var);
            if (observed == observed_target_grads.end() || observed->second.ptr == grads[id].ptr)
                grad = move(grads[id]);
            else
                grad = move(observed->second);
        }
        if (!grad && materialize_grads) {
            // TODO: better warning message
            warn_grad_break(i, var);
            grad = make_number(0.f, var);
            assign_attrs(grad.ptr, var);
        }
    }
    // Record only producers this backward actually differentiated. Boundary
    // inputs of a partial grad are not proof that their own graph was consumed.
    // Snapshot ids before the non-retained graph can release these nodes.
    vector<int64> completed_backward_vars;
    if (!retain_graph && !step_backward_probes.empty()) {
        for (Var* v : gvars) {
            Op* producer = backward_grad_fn(v);
            if (producer && grad_nodes.has(producer))
                completed_backward_vars.push_back(v->id);
        }
    }
    if (!retain_graph) {
        unordered_set<Var*> held_vars;
        held_vars.reserve(runtime_holder_state().holders().size());
        for (auto& vh : runtime_holder_state().holders())
            held_vars.insert(vh->var);
        SetupFreeBuffer setup_free_buffer;
        // Mark before releasing: set_stop_grad drops backward liveness and can
        // queue the node for free. The mark records *why* this var stopped
        // being differentiable -- stop_grad alone cannot say -- so a later
        // backward reaching it can report the released graph instead of
        // returning zeros that look like a legitimate x.stop_grad().
        for (int i=int(gvars.size())-1; i>=0; i--)
            if (!held_vars.count(gvars[i]) && gvars[i]->liveness.backward.active()) {
                gvars[i]->set_flag(VarFlags::_graph_freed);
                gvars[i]->set_stop_grad();
            }
        for (int i=0; i<grads.size(); i++)
            if (grads[i]) {
                grads[i]->set_flag(VarFlags::_graph_freed);
                grads[i]->set_stop_grad();
            }
    }
    for (auto& probe : step_backward_probes)
        probe.variables.insert(completed_backward_vars.begin(), completed_backward_vars.end());
    return results;
}

// Preserve the long-standing three-argument C++ symbol for cached extensions
// and out-of-tree users. The optional binding calls the four-argument overload.
vector<VarPtr> grad(Var* loss, vector<Var*> targets, bool retain_graph) {
    return grad(loss, move(targets), retain_graph, true);
}

} // jittor
