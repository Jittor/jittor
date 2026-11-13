#include "hiprand_random_op.h"
#include "random.h"
#include "core/var.h"
#include "ops/composite/op_capability.h"
#include "runtime/init.h"
#include "runtime/rng_state.h"
#include <map>

namespace jittor {
namespace {
std::map<int, RandomGeneratorState>& device_generators() {
    static std::map<int, RandomGeneratorState> states;
    return states;
}
void reset_default_generators(int) { device_generators().clear(); }
struct SeedRegistration {
    SeedRegistration() { add_set_seed_callback(reset_default_generators); }
} seed_registration;

RandomGeneratorState& default_generator(int device) {
    auto& states = device_generators();
    auto found = states.find(device);
    if (found == states.end())
        found = states.emplace(device, RandomGeneratorState(
            int64(runtime_rng_state().global_seed), device)).first;
    return found->second;
}

RegisterOpCapability<VarPtr, NanoVector, NanoString, NanoString, RandomGenerator*> random(
    BackendId::Rocm, OpCapability::Random, "hiprand_random", nullptr,
    {"float32", "float64"});
} // namespace

HiprandRandomOp::HiprandRandomOp(NanoVector shape, NanoString dtype, NanoString type,
                               RandomGenerator* generator) {
    USER_CHECK(dtype == ns_float32 || dtype == ns_float64)
        << "hiprand_random supports float32 and float64";
    USER_CHECK(type == ns_uniform || type == ns_normal)
        << "hiprand_random supports uniform and normal distributions";
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    output = create_output(shape, dtype);
    normal = type == ns_normal;
    auto& state = generator ? *generator->state : default_generator(output->device_id);
    USER_CHECK(state.device_id == output->device_id)
        << "HIP random generator must belong to the output device";
    const int64 count = output->num;
    int64 values = count;
    if (normal || dtype == ns_float64)
        USER_CHECK(!__builtin_add_overflow(count, count & 1, &values))
            << "HIP random count overflows int64";
    int64 raw;
    USER_CHECK(!__builtin_mul_overflow(values, int64(dtype == ns_float64 ? 2 : 1), &raw))
        << "HIP random stream reservation overflows int64";
    // Reserve at issue time, so lazy execution, reseeding and replay never
    // change the seed/counter span already owned by this draw.
    seed = uint64(state.seed_);
    offset = uint64(state.reserve_cuda(raw));
}

void HiprandRandomOp::run() {
    auto stream = static_cast<hipStream_t>(
        backend_ops(BackendId::Rocm).compute_stream(output->device_id));
    const auto status = hiprand_generate(output->mem_ptr, output->num,
        output->dtype() == ns_float64, normal, seed, offset, stream);
    CHECK(status == hipSuccess) << "HIP random generation: " << hipGetErrorString(status);
}
} // namespace jittor
