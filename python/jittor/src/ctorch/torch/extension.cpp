#include "extension.h"

namespace jittor {

    void AT_CUDA_CHECK(cudaError_t status) { 
                cudaError_t error = status;                                           
        if (error != cudaSuccess) {                                           
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) 
                    << " at line: " << __LINE__ << std::endl;               
            exit(EXIT_FAILURE);                                               
        }    
    }

    #define AT_PRIVATE_CASE_TYPE(enum_type, type, ...) \
        case enum_type: {                                \
            using scalar_t = type;                         \
            return __VA_ARGS__();                          \
        }

    #define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)                 \
    [&] {                                                                      \
        int _st = TYPE; \
        switch (_st) {                                                           \
        AT_PRIVATE_CASE_TYPE(2, double, __VA_ARGS__)      \
        AT_PRIVATE_CASE_TYPE(1, float, __VA_ARGS__)        \
        default:                                                               \
        break; \
        }                                                                        \
    }()



    namespace at {
        namespace cuda {
            cudaStream_t getCurrentCUDAStream() {
                return (cudaStream_t)0;
            }

            OptionalCUDAGuard::OptionalCUDAGuard() {}
            OptionalCUDAGuard::OptionalCUDAGuard(int x) {}
            
        }
    }

    namespace torch {
        
        // definiton of torch kTypes
        int kUInt8 = 0;
        int kFloat = 1;
        int kDouble = 2;
        int kHalf = 3;
        int kFloat16 = 3;
        int kFloat32 = 1;
        int kInt32 = 4;
        int kCUDA = 1;

        Option Option::dtype(int dt) const {
            Option temp = *this;
            temp.dtype_ = dt;
            return temp;
        }
        
        Option Option::device(int device, int devid=0) const {
            Option temp = *this;
            temp.device_ = device;
            temp.devid_ = devid;
            return temp;
        }

        Option::Option(int dt):dtype_(dt) {}
        Option::Option() {}
        
        int Device::index() {
            return 0;
        }

        int Device::type() {
            return use_cuda;
        }

        int Option::type() {
            return use_cuda;
        }

        bool Device::operator ==(const Device& b) const { return true; }

        Tensor::Tensor():format(at::MemoryFormat::Contiguous),jtptr(nullptr),ndim(0) {}

        Tensor::Tensor(VarPtr& ptr) {
            jtptr = new VarHolder(ptr.ptr);
            ndim = jtptr->shape().size();
        }

        Tensor::Tensor(const Tensor& b) {
            if(!b.jtptr) jtptr = nullptr;
            else
                jtptr = new VarHolder(b.jtptr->var);
        }
        
        NanoVector Tensor::size() {
            if(!jtptr) return 0;
            return jtptr->shape();
        }

        NanoVector Tensor::sizes() {
            if(!jtptr) return 0;
            return jtptr->shape();
        }

        int Tensor::size(int i) {
            if(!jtptr) {
                LOGir << "Tensor is None.";
                return -1;
            }
            if(i == -1) 
                i = jtptr->shape().size() - 1;
            return jtptr->shape()[i];
        }

        int Tensor::numel() {
            if(!jtptr)
                return 0;
            return jtptr->numel();
        }

        void Tensor::init_stride() {
            if(!jtptr) return;
            int64 prod = 1;
            for(int i=jtptr->shape().size()-1;i>=0;i--) {
                strides.push_back(prod);
                prod *= jtptr->shape()[i];
            }
        }

        NanoVector Tensor::stride(){
            if(strides.size() == 0) init_stride();
            return strides;
        } 

        int Tensor::stride(int i) {
            if(!jtptr) {
                LOGir << "Tensor is None.";
                return -1;
            }
            if(strides.size() == 0) init_stride();
            if(i == -1)
                i = 0;
            return strides[strides.size() - i - 1];
        }

        Tensor contiguous() {
            return *this;
        }

        int get_device() {
            return 0;
        }

        bool defined() {
            if(!jtptr) return 0;
            if(!jtptr->var) return 0;
            return 1;
        }

        int Tensor::dtype() const {
            if(!jtptr) return -1; // nullptr
            NanoString dtype_ = jtptr->dtype();
            if(dtype_ == "uint8") 
                return 0;
            if(dtype_ == "float16")
                return 3;
            if(dtype_ == "float32")
                return 1;
            if(dtype_ == "float64")
                return 2;
            if(dtype_ == "int32")
                return 4;
            return -1; // non-exist types
        }

        int Tensor::scalar_type() {
            return dtype();
        }

        at::MemoryFormat Tensor::suggest_memory_format() {
            return format;
        }
    
        void Tensor::sync(bool device_sync) {
            if(jtptr)
                jtptr->sync(device_sync);
        }

        int Tensor::dim() {
            if(!jtptr)
                return 0;
            return jtptr->shape().size();
        }

        bool Tensor::is_contiguous() const {
            return true;
        }

        Device Tensor::device() { // device is controlled by jittor 
            return Device(); // so all tensors are on the same device.
        }

        void Tensor::cuda() {
            return;
        }
        
        bool Tensor::is_cuda() {
            return use_cuda;
        }

        Option Tensor::options() { // assume that jtptr is not nullptr
            return Option(dtype());
        }

        static auto make_empty = get_op_info("empty").get_constructor<VarPtr, NanoVector, NanoString>();
        static auto make_number = get_op_info("number").get_constructor<VarPtr, float, Var*>();
        
        Tensor zeros_like(Tensor& refer_tensor) {
            Varptr ptr = make_number(0, refer_tensor.jtptr->var);
            return Tensor(ptr);
        }

        Tensor empty_like(Tensor& refer_tensor) {
            VarPtr ptr = make_empty(refer_tensor.size(), refer_tensor.dtype());
            return Tensor(ptr);
        }

        Tensor empty(NanoVector shape, Option option, at::MemoryFormat format) {
            // todo: add special support for different formats. For now all outputs are contiguous format.
            VarPtr ptr;

            switch(option.dtype_) {
                case 0:
                    ptr = make_empty(shape, "uint8"); 
                    break;
                case 1:
                    ptr = make_empty(shape, "float32");
                    break;
                case 2:
                    ptr = make_empty(shape, "float64");
                    break;
                case 4:
                    ptr = make_empty(shape, "int32");
                    break;
                default:
                    ptr = make_empty(shape, "float32");
                    break;  
            }
            return Tensor(ptr);
        }

        Option TensorOptions() {
            return Option();
        }

        // void test_tensor(Tensor a) {
        //     // todo: add special support for different formats.
        //     std::cout << "Success!!" << std::endl;
        // }

        // Tensor test_ret_tensor(Tensor a) {
        //     return a;
        // }
    }
    
    namespace at {
        // in face at::Tensor is not differentiable. TODO: set requires_grad to false for it.
        using Tensor = torch::Tensor; 
    }

    int device_of(torch::Tensor a) {
        if(use_cuda) 
            return 1;
        else return 0;
    }
}

namespace pybind11 { namespace detail {

// PYBIND11_TYPE_CASTER(jittor::torch::Tensor, const_name("Tensor"));

bool type_caster<jittor::torch::Tensor>::load(handle src, bool) {
    PyObject *source = src.ptr();
    if(source != Py_None) {
        jittor::VarHolder* var_holder = jittor::from_py_object<jittor::VarHolder*>(source);
        if (!var_holder)
            return false;
        value.jtptr = var_holder;
    } else {
        value.jtptr = nullptr;
    }
    return !PyErr_Occurred();
}

handle type_caster<jittor::torch::Tensor>::cast(jittor::torch::Tensor src, return_value_policy, handle) {
    jittor::PyObjHolder obj(_PyObject_New(&jittor::PyjtVarHolder.ht_type));
    auto ptr = GET_RAW_PTR(jittor::VarHolder*, obj.obj);
    new (ptr) jittor::VarHolder (src.jtptr->var);
    return obj.release();
}

}} 