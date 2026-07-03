// ============================================================================
// torch/extension.h  —  Jittor-backed libtorch C++ extension ABI shim
// ----------------------------------------------------------------------------
// Lets a PyTorch C++/CUDA extension (one that #include <torch/extension.h>, uses
// torch::Tensor as a data container, and PYBIND11_MODULE) compile and run on
// Jittor with ZERO dependency on real libtorch.  torch::Tensor wraps a
// jittor::VarHolder*; the Python jittor.Var <-> Tensor bridge uses jittor's own
// pyjt converters.
//
// IMPORTANT: this public header includes NO jittor or CUDA headers.  Jittor's
// headers transitively pull <cuda_runtime.h> (vector_types.h -> float3/int3),
// which clashes with extensions that define their own host-side float3.  So the
// VarHolder is forward-declared here and every jittor/CUDA-touching operation is
// out-of-line in jtorch_aten.cu (compiled by nvcc, linked into each ext .so).
// ============================================================================
#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <initializer_list>
#include <memory>
#include <algorithm>
#include <functional>
#include <utility>
#include <tuple>
#include <map>
#include <unordered_map>
#include <limits>
#include <optional>
#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;   // torch/extension.h exposes this alias for exts

namespace jittor { struct VarHolder; }   // forward — no jittor headers pulled

namespace jtorch {

// torch/at use c10::optional (== std::optional here); exts write at::optional<T>.
template <typename T> using optional = std::optional<T>;
using std::nullopt;

// ------------------------- dtype -------------------------------------------
enum class ScalarType : int8_t {
    Byte = 0, Char = 1, Short = 2, Int = 3, Long = 4,
    Half = 5, Float = 6, Double = 7,
    ComplexHalf = 8, ComplexFloat = 9, ComplexDouble = 10,
    Bool = 11, UInt16 = 27, UInt32 = 28, UInt64 = 29, Undefined = 127,
};

struct Dtype {
    ScalarType st;
    Dtype(ScalarType s = ScalarType::Undefined) : st(s) {}
    operator ScalarType() const { return st; }
    bool operator==(const Dtype& o) const { return st == o.st; }
    bool operator!=(const Dtype& o) const { return st != o.st; }
    bool operator==(ScalarType o) const { return st == o; }
    bool operator!=(ScalarType o) const { return st != o; }
};

enum class DeviceType : int8_t { CPU = 0, CUDA = 1 };

struct Device {
    DeviceType type_; int index_;
    Device(DeviceType t = DeviceType::CUDA, int idx = 0) : type_(t), index_(idx) {}
    DeviceType type() const { return type_; }
    int index() const { return index_; }
    bool is_cuda() const { return type_ == DeviceType::CUDA; }
    bool is_cpu()  const { return type_ == DeviceType::CPU; }
    bool operator==(const Device& o) const { return type_ == o.type_; }
};

struct TensorOptions {
    ScalarType dtype_ = ScalarType::Float;
    DeviceType device_ = DeviceType::CUDA;
    bool has_dtype_ = false, has_device_ = false;
    TensorOptions() {}
    TensorOptions(ScalarType s) : dtype_(s), has_dtype_(true) {}
    TensorOptions(Dtype d) : dtype_(d.st), has_dtype_(true) {}
    TensorOptions(DeviceType d) : device_(d), has_device_(true) {}
    TensorOptions(Device d) : device_(d.type_), has_device_(true) {}
    TensorOptions dtype(ScalarType s) const { TensorOptions o=*this; o.dtype_=s; o.has_dtype_=true; return o; }
    TensorOptions dtype(Dtype d) const { return dtype(d.st); }
    TensorOptions device(DeviceType d, int=0) const { TensorOptions o=*this; o.device_=d; o.has_device_=true; return o; }
    TensorOptions device(Device d) const { return device(d.type_); }
    TensorOptions requires_grad(bool) const { return *this; }
    TensorOptions pinned_memory(bool) const { return *this; }
    ScalarType dtype() const { return dtype_; }
    DeviceType device() const { return device_; }
};
inline TensorOptions dtype(ScalarType s) { return TensorOptions(s); }
inline TensorOptions dtype(Dtype d) { return TensorOptions(d.st); }
inline TensorOptions device(DeviceType d) { return TensorOptions(d); }
inline TensorOptions device(Device d) { return TensorOptions(d); }

struct IntArrayRef {
    std::vector<int64_t> v;
    IntArrayRef() {}
    // single-element ctor — libtorch's ArrayRef has `ArrayRef(const T& OneElt)`,
    // so exts write torch::empty(0) / torch::full(N, ...) with a bare int meaning
    // a 1-D shape {N} (fused-ssim does `torch::empty(0)`).
    IntArrayRef(int64_t x) : v{x} {}
    IntArrayRef(std::initializer_list<int64_t> l) : v(l) {}
    IntArrayRef(const std::vector<int64_t>& x) : v(x) {}
    IntArrayRef(const int64_t* p, size_t n) : v(p, p + n) {}
    size_t size() const { return v.size(); }
    int64_t operator[](size_t i) const { return v[i]; }
    const int64_t* data() const { return v.data(); }
    auto begin() const { return v.begin(); }
    auto end() const { return v.end(); }
    std::vector<int64_t> vec() const { return v; }
    bool equals(const IntArrayRef& o) const { return v == o.v; }
};
// shape equality (nvdiffrast compares tensor.sizes() == std::vector<int64_t>)
inline bool operator==(const IntArrayRef& a, const IntArrayRef& b) { return a.equals(b); }
inline bool operator!=(const IntArrayRef& a, const IntArrayRef& b) { return !a.equals(b); }

// Generic ArrayRef<T> (exts pass at::ArrayRef<at::Tensor>{a,b} to check helpers).
// Non-owning view; constructed from initializer_list / vector / pointer+len.
template <typename T>
struct ArrayRef {
    const T* p_ = nullptr; size_t n_ = 0;
    std::vector<T> own_;                       // backs initializer_list/vector ctors
    ArrayRef() {}
    ArrayRef(std::initializer_list<T> l) : own_(l) { p_ = own_.data(); n_ = own_.size(); }
    ArrayRef(const std::vector<T>& x) : own_(x) { p_ = own_.data(); n_ = own_.size(); }
    ArrayRef(const T* p, size_t n) : p_(p), n_(n) {}
    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    const T& operator[](size_t i) const { return p_[i]; }
    const T* begin() const { return p_; }
    const T* end() const { return p_ + n_; }
};

// ------------------------- out-of-line bridge (defined in jtorch_aten.cu) ----
class Tensor;
namespace detail {
    void*    vh_device_ptr(jittor::VarHolder*);        // device ptr, no cpu-migrate
    int64_t  vh_ndim(jittor::VarHolder*);
    int64_t  vh_size(jittor::VarHolder*, int64_t d);
    void     vh_shape(jittor::VarHolder*, std::vector<int64_t>& out);
    int64_t  vh_numel(jittor::VarHolder*);
    int64_t  vh_dsize(jittor::VarHolder*);
    const char* vh_dtype_name(jittor::VarHolder*);     // "float32" etc.
    bool     vh_is_cuda(jittor::VarHolder*);
    int      vh_device_type(jittor::VarHolder*);        // 0=CPU, 1=CUDA (no migrate)
    double   vh_item_double(jittor::VarHolder*);
    int64_t  vh_item_int(jittor::VarHolder*);

    Tensor   adopt(jittor::VarHolder* vh, bool owns);  // wrap (owns => deleting)
    jittor::VarHolder* clone_holder(jittor::VarHolder* vh);  // fresh holder, same Var

    // python bridge
    bool     is_jittor_var(void* pyobj);
    bool     pyvar_is_ext_mutable(void* pyobj);
    Tensor   tensor_from_pyvar(void* pyobj);           // borrow
    void     commit_tensor_to_pyvar(void* pyobj, const Tensor& t);
    void*    tensor_to_pyvar(const Tensor& t);         // +1 PyObject*

    ScalarType name_to_scalar(const char* n);
}

// ------------------------- Tensor ------------------------------------------
class Tensor {
public:
    std::shared_ptr<jittor::VarHolder> vh_;   // type-erased deleter (incomplete-OK)
    Tensor() {}
    explicit Tensor(std::shared_ptr<jittor::VarHolder> vh) : vh_(std::move(vh)) {}

    bool defined() const { return (bool)vh_; }
    jittor::VarHolder* _vh() const { return vh_.get(); }

    int64_t dim() const { return detail::vh_ndim(vh_.get()); }
    int64_t ndimension() const { return dim(); }
    int64_t numel() const { return detail::vh_numel(vh_.get()); }
    int64_t nbytes() const { return detail::vh_numel(vh_.get()) * detail::vh_dsize(vh_.get()); }
    int64_t element_size() const { return detail::vh_dsize(vh_.get()); }
    int64_t size(int64_t d) const { return detail::vh_size(vh_.get(), d); }
    IntArrayRef sizes() const { IntArrayRef r; detail::vh_shape(vh_.get(), r.v); return r; }
    int64_t stride(int64_t d) const {
        int64_t nd = dim(); if (d < 0) d += nd; int64_t s = 1;
        for (int64_t i = nd - 1; i > d; --i) s *= size(i); return s;
    }
    ScalarType scalar_type() const { return detail::name_to_scalar(detail::vh_dtype_name(vh_.get())); }
    Dtype dtype() const { return Dtype(scalar_type()); }
    Device device() const {
        return Device(detail::vh_device_type(vh_.get()) ? DeviceType::CUDA : DeviceType::CPU);
    }
    int64_t get_device() const { return detail::vh_device_type(vh_.get()) ? 0 : -1; }
    bool is_cuda() const { return detail::vh_device_type(vh_.get()) != 0; }
    // torch's Tensor::is_cpu() — native exts (nvdiffrast NVDR_CHECK_CPU, cumesh
    // xatlas) check host-residency this way. Mirrors device().is_cpu(): reads the
    // Var's actual allocator residency (0=CPU) without force-migrating.
    bool is_cpu() const { return detail::vh_device_type(vh_.get()) == 0; }
    bool is_contiguous() const { return true; }
    Tensor contiguous() const { return *this; }
    Tensor detach() const { return *this; }   // no autograd graph in this shim
    TensorOptions options() const {
        return TensorOptions(scalar_type()).device(is_cuda() ? DeviceType::CUDA : DeviceType::CPU);
    }

    template <typename T = void>
    T* data_ptr() const { return reinterpret_cast<T*>(detail::vh_device_ptr(vh_.get())); }
    void* data_ptr_void() const { return detail::vh_device_ptr(vh_.get()); }
    // legacy alias: many exts (diff-gaussian-rasterization, simple-knn, fused-ssim)
    // still call .data<T>() (the pre-1.5 spelling of data_ptr<T>()).
    template <typename T = float>
    T* data() const { return reinterpret_cast<T*>(detail::vh_device_ptr(vh_.get())); }

    template <typename T> T item() const {
        // exts use .item<int>()/.item<float>(); magnitudes fit double.
        return (T)detail::vh_item_double(vh_.get());
    }

    Tensor view(IntArrayRef shape) const;
    Tensor view(ScalarType st) const;          // bit-reinterpret to same-width dtype
    Tensor reshape(IntArrayRef shape) const { return view(shape); }
    Tensor clone() const;
    // in-place reallocation (diff-gaussian-rasterization's resizeFunctional grows
    // empty byte buffers to the size the rasterizer computes): swaps the underlying
    // Var for a fresh empty one of `shape`, same dtype + device residency. Returns
    // *this so `t.resize_({N}).data_ptr()` chains.
    Tensor& resize_(IntArrayRef shape);
    Tensor operator[](int64_t index) const;   // row i along dim 0 (copy)

    // method-form ATen ops (defined in jtorch_aten.cu; wired on first real use)
    Tensor cumsum(int64_t dim) const;
    std::tuple<Tensor, Tensor> sort(int64_t dim = -1, bool descending = false) const;
    Tensor masked_select(const Tensor& mask) const;
    Tensor ne(const Tensor& other) const;
    Tensor ne_scalar(double other) const;      // elementwise self != scalar -> bool mask
};

// elementwise self != scalar -> bool mask (exts compare uint keys to sentinel max)
template <typename S, typename = typename std::enable_if<std::is_arithmetic<S>::value>::type>
inline Tensor operator!=(const Tensor& self, S other) { return self.ne_scalar((double)other); }
inline Tensor operator!=(const Tensor& a, const Tensor& b) { return a.ne(b); }

// device_of(t) -> optional<Device>; nvdiffrast feeds this to OptionalCUDAGuard.
inline optional<Device> device_of(const Tensor& t) {
    if (!t.defined()) return nullopt;
    return t.device();
}

using at_Tensor = Tensor;

// ------------------------- factories (defined in jtorch_aten.cu) ------------
Tensor empty(IntArrayRef size, TensorOptions opt = {});
inline Tensor empty(IntArrayRef size, ScalarType st) { return empty(size, TensorOptions(st)); }
Tensor empty_like(const Tensor& t);
Tensor empty_like(const Tensor& t, TensorOptions opt);
Tensor zeros(IntArrayRef size, TensorOptions opt = {});
inline Tensor zeros(IntArrayRef size, ScalarType st) { return zeros(size, TensorOptions(st)); }
Tensor zeros_like(const Tensor& t);
Tensor zeros_like(const Tensor& t, TensorOptions opt);
Tensor full(IntArrayRef size, double value, TensorOptions opt = {});
inline Tensor full(IntArrayRef size, double value, ScalarType st) { return full(size, value, TensorOptions(st)); }
Tensor ones(IntArrayRef size, TensorOptions opt = {});
inline Tensor ones(IntArrayRef size, ScalarType st) { return ones(size, TensorOptions(st)); }
Tensor from_blob(const void* data, IntArrayRef size, TensorOptions opt = {});
inline Tensor from_blob(const void* data, IntArrayRef size, ScalarType st) { return from_blob(data, size, TensorOptions(st)); }

// a few ATen ops referenced by these exts (defined in jtorch_aten.cu)
Tensor cumsum(const Tensor& self, int64_t dim);
Tensor cumsum(const Tensor& self, int64_t dim, ScalarType out_dtype);
Tensor argsort(const Tensor& self, int64_t dim = -1, bool descending = false);
std::tuple<Tensor, Tensor> sort(const Tensor& self, int64_t dim = -1, bool descending = false);
Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index);
// at::_unique(self, sorted=true, return_inverse=false) -> (unique_values, inverse_indices)
std::tuple<Tensor, Tensor> _unique(const Tensor& self, bool sorted = true, bool return_inverse = false);

} // namespace jtorch

// libtorch's global log-level flag (nvdiffrast's get/set_log_level read/write it).
inline int FLAGS_caffe2_log_level = 1;

// ============================================================================
// expose under torch:: / at:: / c10::
// ============================================================================
namespace c10 {
    using ScalarType = jtorch::ScalarType; using Device = jtorch::Device; using DeviceType = jtorch::DeviceType;
    template <typename T> using optional = jtorch::optional<T>;
    using jtorch::nullopt;
}
namespace at {
    using Tensor = jtorch::Tensor; using jtorch::IntArrayRef; using jtorch::TensorOptions;
    template <typename T> using ArrayRef = jtorch::ArrayRef<T>;
    using ScalarType = jtorch::ScalarType; using Device = jtorch::Device; using DeviceType = jtorch::DeviceType;
    template <typename T> using optional = jtorch::optional<T>;
    using jtorch::nullopt;
    using jtorch::empty; using jtorch::zeros; using jtorch::ones; using jtorch::full;
    using jtorch::empty_like; using jtorch::zeros_like; using jtorch::from_blob;
    using jtorch::dtype; using jtorch::device; using jtorch::device_of;
    using jtorch::cumsum; using jtorch::argsort; using jtorch::sort; using jtorch::index_select;
    using jtorch::_unique;
    // ScalarType constants (exts use at::kInt / at::kFloat / ...)
    constexpr ScalarType kByte=ScalarType::Byte, kChar=ScalarType::Char, kShort=ScalarType::Short;
    constexpr ScalarType kInt=ScalarType::Int, kLong=ScalarType::Long, kHalf=ScalarType::Half;
    constexpr ScalarType kFloat=ScalarType::Float, kDouble=ScalarType::Double, kBool=ScalarType::Bool;
    constexpr ScalarType kUInt16=ScalarType::UInt16, kUInt32=ScalarType::UInt32, kUInt64=ScalarType::UInt64;
    constexpr DeviceType kCUDA=DeviceType::CUDA, kCPU=DeviceType::CPU;
}
namespace torch {
    using jtorch::Tensor; using jtorch::IntArrayRef; using jtorch::TensorOptions;
    template <typename T> using ArrayRef = jtorch::ArrayRef<T>;
    using jtorch::Device; using jtorch::Dtype;
    using ScalarType = jtorch::ScalarType; using DeviceType = jtorch::DeviceType;
    template <typename T> using optional = jtorch::optional<T>;
    using jtorch::nullopt; using jtorch::device_of;

    constexpr ScalarType kByte=ScalarType::Byte,   kUInt8=ScalarType::Byte;
    constexpr ScalarType kChar=ScalarType::Char,   kInt8=ScalarType::Char;
    constexpr ScalarType kShort=ScalarType::Short, kInt16=ScalarType::Short;
    constexpr ScalarType kInt=ScalarType::Int,     kInt32=ScalarType::Int;
    constexpr ScalarType kLong=ScalarType::Long,   kInt64=ScalarType::Long;
    constexpr ScalarType kHalf=ScalarType::Half,   kFloat16=ScalarType::Half;
    constexpr ScalarType kFloat=ScalarType::Float, kFloat32=ScalarType::Float;
    constexpr ScalarType kDouble=ScalarType::Double, kFloat64=ScalarType::Double;
    constexpr ScalarType kBool=ScalarType::Bool;
    constexpr ScalarType kUInt16=ScalarType::UInt16, kUInt32=ScalarType::UInt32, kUInt64=ScalarType::UInt64;
    constexpr DeviceType kCUDA=DeviceType::CUDA, kCPU=DeviceType::CPU;

    using jtorch::empty; using jtorch::empty_like; using jtorch::zeros; using jtorch::zeros_like;
    using jtorch::ones; using jtorch::full; using jtorch::from_blob;
    using jtorch::dtype; using jtorch::device;
    using jtorch::cumsum; using jtorch::argsort; using jtorch::sort; using jtorch::index_select;
    using jtorch::_unique;
}

// ---- macros ----------------------------------------------------------------
namespace jtorch { namespace detail {
template <typename... Ts> inline std::string _concat(Ts&&... ts) {
    std::ostringstream os; (void)std::initializer_list<int>{((os << ts), 0)...}; return os.str();
}
}}
// Brace-enclosed form (NOT do/while): usable both as a statement `TORCH_CHECK(x);`
// AND directly before a `}` — nvdiffrast's NVDR_CHECK wraps it as
// `do { TORCH_CHECK(...) } while(0)` with no inner semicolon, so the body must be
// a self-terminating block (the inner `;` after throw stays inside the braces).
#define TORCH_CHECK(cond, ...) { if (!(cond)) throw std::runtime_error(jtorch::detail::_concat("TORCH_CHECK failed: ", #cond, " ", ##__VA_ARGS__)); }
#define TORCH_INTERNAL_ASSERT(cond, ...) TORCH_CHECK(cond, ##__VA_ARGS__)
#define AT_ASSERTM(cond, ...) TORCH_CHECK(cond, ##__VA_ARGS__)
#define AT_ASSERT(cond) TORCH_CHECK(cond)
#define AT_ERROR(...) do { throw std::runtime_error(jtorch::detail::_concat(__VA_ARGS__)); } while (0)
#define TORCH_WARN(...) ((void)0)

// ---- minimal glog/c10-style logging (nvdiffrast uses `LOG(INFO) << ...`) -----
// A sink that swallows the stream and, for INFO/WARNING, prints to stderr only if
// the message severity passes FLAGS_caffe2_log_level (info=0,warn=1,err=2,fatal=3).
namespace jtorch { namespace detail {
enum LogSeverity { kLOG_INFO = 0, kLOG_WARNING = 1, kLOG_ERROR = 2, kLOG_FATAL = 3 };
struct LogSink {
    int sev_; std::ostringstream os_;
    explicit LogSink(int sev) : sev_(sev) {}
    template <typename T> LogSink& operator<<(const T& v) { os_ << v; return *this; }
    ~LogSink() noexcept(false) {
        if (sev_ >= FLAGS_caffe2_log_level) std::cerr << os_.str() << std::endl;
        if (sev_ >= kLOG_FATAL) throw std::runtime_error(os_.str());
    }
};
}} // namespace jtorch::detail
#ifndef INFO
#define INFO    (::jtorch::detail::kLOG_INFO)
#define WARNING (::jtorch::detail::kLOG_WARNING)
#define ERROR   (::jtorch::detail::kLOG_ERROR)
#define FATAL   (::jtorch::detail::kLOG_FATAL)
#endif
#ifndef LOG
#define LOG(sev) (::jtorch::detail::LogSink((int)(sev)))
#endif

// ---- pybind11 type_caster: Python jittor.Var <-> torch::Tensor -------------
namespace pybind11 { namespace detail {
template <> struct type_caster<jtorch::Tensor> {
    PYBIND11_TYPE_CASTER(jtorch::Tensor, _("Tensor"));
    object keepalive;
    bool mutable_arg = false;
    bool load(handle src, bool) {
        if (!src || src.is_none()) return false;
        if (!jtorch::detail::is_jittor_var(src.ptr())) return false;
        mutable_arg = jtorch::detail::pyvar_is_ext_mutable(src.ptr());
        value = jtorch::detail::tensor_from_pyvar(src.ptr());
        keepalive = reinterpret_borrow<object>(src);
        return true;
    }
    ~type_caster() {
        if (mutable_arg && keepalive && value.defined())
            jtorch::detail::commit_tensor_to_pyvar(keepalive.ptr(), value);
    }
    static handle cast(const jtorch::Tensor& t, return_value_policy, handle) {
        if (!t.defined()) { Py_RETURN_NONE; }
        return handle((PyObject*)jtorch::detail::tensor_to_pyvar(t));
    }
};
}} // namespace pybind11::detail
