# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
from jittor_utils import env_or_try_find
import jittor_utils
import ctypes
import glob
import jittor.compiler as compiler
import jittor as jt

has_acl = 0
cc_flags = ""
tikcc_path = env_or_try_find('tikcc_path', 'ccec')
dlopen_flags = os.RTLD_NOW | os.RTLD_GLOBAL
compiler.has_acl = has_acl

# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/aoe/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub:/usr/local/Ascend/ascend-toolkit/latest/tools/tikicpulib/lib/Ascend910A:/usr/local/Ascend/ascend-toolkit/latest/toolkit/tools/simulator/Ascend910A/lib:/opt/AXESMI/lib64:/usr/local/Ascend/driver/lib64/driver/
# export PYTHONPATH=/home/cjld/new_jittor/jittor/python
# export tikcc_path=g++

# conda activate cann
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# export PYTHONPATH=/home/cjld/new_jittor/jittor/python:/home/cjld/new_jittor/jittor/my/jtorch/python:$PYTHONPATH
# export TASK_QUEUE_ENABLE=0
# python3 -m jittor.test.test_acl -k array
# jittor: conda activate cann && source /usr/local/Ascend/ascend-toolkit/set_env.sh && PYTHONPATH=/home/cjld/new_jittor/jittor/python:/home/cjld/new_jittor/jittor/my/jtorch/python:$PYTHONPATH && cd /home/cjld/new_jittor/jittor/my/mm_benchmark
# python3 -m jittor.test.test_acl -k test_sum
# export ASCEND_SLOG_PRINT_TO_STDOUT=0
# ASCEND_GLOBAL_LOG_LEVEL
# export DUMP_GE_GRAPH=1
# export DUMP_GRAPH_LEVEL=1

# build pytorch-npu
# bash ./ci/build.sh
# python3 -m pip install ./dist/torch_npu-1.11.0.post1-cp37-cp37m-linux_x86_64.whl  --force-reinstall
# pytorch: conda activate cann && source /usr/local/Ascend/ascend-toolkit/set_env.sh && export TASK_QUEUE_ENABLE=0  && cd /home/cjld/new_jittor/jittor/my/mm_benchmark
# python3 ./mm_bench_pt_npu.py


def install():
    import jittor.compiler as compiler
    global has_acl, cc_flags
    acl_compiler_home = os.path.dirname(__file__)
    cc_files = sorted(glob.glob(acl_compiler_home + "/**/*.cc",
                                recursive=True))
    cc_files2 = []
    for name in cc_files:
        if "acl_op_exec" in name:
            compiler.extra_core_files.append(name)
        else:
            cc_files2.append(name)
    cc_files = cc_files2
    ascend_toolkit_home = os.getenv('ASCEND_TOOLKIT_HOME')
    cc_flags += f" -DHAS_CUDA -DIS_ACL  \
    -I{ascend_toolkit_home}/include/ \
    -L{ascend_toolkit_home}/lib64/ \
    -I{acl_compiler_home} -lascendcl -lacl_op_compiler "

    ctypes.CDLL("libascendcl.so", dlopen_flags)
    f'''
    -ltikc_runtime
    -I/usr/local/Ascend/driver/include/ \
    -L{ascend_toolkit_home}/compiler/lib64/ \
    -L{ascend_toolkit_home}/runtime/lib64/ \
    '''
    jittor_utils.LOG.i("ACL detected")

    global mod
    mod = jittor_utils.compile_module(
        '''
#include "common.h"
namespace jittor {
// @pyjt(process)
string process_acl(const string& src, const string& name, const map<string,string>& kargs);
// @pyjt(init_acl_ops)
void init_acl_ops();
}''', compiler.cc_flags + " " + " ".join(cc_files) + cc_flags)
    jittor_utils.process_jittor_source("acl", mod.process)

    has_acl = 1
    os.environ["use_mkl"] = "0"
    compiler.setup_fake_cuda_lib = True


def install_extern():
    return False


def check():
    import jittor.compiler as compiler
    global has_acl, cc_flags
    if tikcc_path:
        try:
            install()
        except Exception as e:
            jittor_utils.LOG.w(f"load ACL failed, exception: {e}")
            has_acl = 0
    compiler.has_acl = has_acl
    compiler.tikcc_path = tikcc_path
    if not has_acl: return False
    compiler.cc_flags += cc_flags
    compiler.nvcc_path = tikcc_path
    compiler.nvcc_flags = compiler.cc_flags.replace("-std=c++14", "")
    return True


def post_process():
    if has_acl:
        from jittor import pool
        pool.pool_use_code_op = False
        import jittor as jt
        jt.flags.use_cuda_host_allocator = 1
        jt.flags.use_parallel_op_compiler = 0
        jt.flags.amp_reg |= 32 + 4  # 32 keep float16, 4 keep reduce type
        mod.init_acl_ops()


def acl_cmd(name: str, inputs: list, output_dtypes: list, output_shapes: list,
            attr: dict):
    nchw_op = ['MaxPoolWithArgmaxV1', 'MaxPoolGradWithArgmaxV1', 'AvgPoolV2']
    attr_op = [
        'MaxPoolWithArgmaxV1', 'MaxPoolGradWithArgmaxV1', 'AvgPoolV2',
        'AdaptiveAvgPool2d', 'AdaptiveAvgPool2dGrad', 'ReverseV2'
    ]

    input_code = ''
    for i in range(len(inputs)):
        if name in nchw_op:
            input_code += f"op.add(in{i}, true, ACL_FORMAT_NCHW);\n"
        else:
            input_code += f"op.add(in{i}, true);\n"

    output_code = ''
    for i in range(len(output_dtypes)):
        if name in nchw_op:
            output_code += f"op.add(out{i}, false, ACL_FORMAT_NCHW);\n"
        else:
            output_code += f"op.add(out{i}, false);\n"

    # add attr to op
    attr_code = ''
    if name in attr_op:
        for k, v in attr.items():
            if isinstance(v, bool):
                if v == True:
                    attr_code += f"op.set_attr(\"{k}\", 1, 1);\n"
                else:
                    attr_code += f"op.set_attr(\"{k}\", 1, 0);\n"
            elif isinstance(v, str):
                attr_code += f"op.set_attr(\"{k}\", \"{v}\");\n"
            elif k == 'divisor_override_value':
                attr_code += f"op.set_attr(\"{k}\", int64_t({v}), 0);\n"
            else:
                v = str(v).replace('[', '{').replace(']', '}')
                attr_code += f"op.set_attr(\"{k}\", vector<int64_t>{v});\n"
    else:
        for k, v in attr.items():
            if isinstance(v, bool):
                if v == True:
                    attr_code += f"op.set_attr(\"{k}\", 1, 1);\n"
                else:
                    attr_code += f"op.set_attr(\"{k}\", 1, 0);\n"
            elif isinstance(v, str):
                attr_code += f"op.set_attr(\"{k}\", \"{v}\");\n"
            else:
                attr_code += f"op.set_attr(\"{k}\", int({v}));\n"

    #print("input_code",input_code)
    #print("attr_code",attr_code)
    import jittor as jt
    return jt.code(output_shapes,
                   output_dtypes,
                   inputs,
                   cuda_header="""
    #include <acl/acl.h>
    #include <acl/acl_op_compiler.h>
    #include <Python.h>
    #include <pystate.h>

    namespace jittor {
    
    void printDeviceData(const vector<aclTensorDesc*>& output_desc, const vector<aclDataBuffer*>& output_data, const string& name = "", bool input=true) {
        LOGir << "name: " << name;
        if(input)
            LOGir << "is input";
        else
            LOGir << "is ouput";
        for (size_t i = 0; i < output_desc.size(); ++i) {
            void* base_addr = aclGetDataBufferAddr(output_data[i]);
            LOGir << "addr of data[" << i << "] :" << base_addr;
            size_t num_dims = aclGetTensorDescNumDims(output_desc[i]);
            size_t total_size = 1;
            std::vector<int64_t> dims(num_dims);
            
            std::cout << "shape of data: ";
            for (size_t j = 0; j < num_dims; ++j) {
                aclGetTensorDescDimV2(output_desc[i], j, &dims[j]);
                total_size *= dims[j];
                std::cout << dims[j] << ", ";
            }
            int evey_batch_size = total_size/dims[0];
            std::cout << std::endl;

            // for(int i= 0; i < dims[0]; i++) {
            //     evey_batch_size = 16;
            //     std::vector<float> host_buffer(evey_batch_size);
            //     void* offset_addr = static_cast<char*>(base_addr) + i * evey_batch_size * sizeof(float);
            //     aclrtMemcpy(host_buffer.data(), evey_batch_size * sizeof(float), offset_addr, evey_batch_size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            //     std::cout << "batch[" << i << "]:";
            //     for (size_t k = 0; k < evey_batch_size; ++k) {
            //         std::cout << host_buffer[k] << ", ";
            //     }
            //     std::cout << std::endl;
            // }
        }
    }

    struct AclOpRunner {
        string name;
        vector<aclTensorDesc*> input_desc;
        vector<aclTensorDesc*> output_desc;
        vector<aclDataBuffer*> input_data;
        vector<aclDataBuffer*> output_data;
        aclopAttr *attr;
        vector<vector<uint64>> input_host;
        vector<vector<int>> input_host_32;

        AclOpRunner(const string& name) : name(name) {
            attr = aclopCreateAttr();
        }

        ~AclOpRunner() {
            for (auto i : input_desc) aclDestroyTensorDesc(i);
            for (auto i : output_desc) aclDestroyTensorDesc(i);
            for (auto i : input_data) aclDestroyDataBuffer(i);
            for (auto i : output_data) aclDestroyDataBuffer(i);
            aclopDestroyAttr(attr);
        }

        aclDataType get_dtype(NanoString s) {
            if (s == ns_float32) return ACL_FLOAT;
            if (s == ns_float16) return ACL_FLOAT16;
            if (s == ns_int64) return ACL_INT64;
            if (s == ns_int32) return ACL_INT32;
            if (s == ns_int8) return ACL_INT8;
            if (s == ns_int16) return ACL_INT16;
            if (s == ns_uint8) return ACL_UINT8;
            if (s == ns_uint16) return ACL_UINT16;
            if (s == ns_uint32) return ACL_UINT32;
            if (s == ns_bool) return ACL_BOOL;
            LOGf << "Not supported dtype: " << s;
            return ACL_FLOAT;
        }

        void add(Var* v, bool is_input, int format=ACL_FORMAT_ND) {
            int64_t shape[v->shape.size()];
            for (int i=0; i<v->shape.size(); i++) shape[i] = v->shape[i];

            auto desc = aclCreateTensorDesc(get_dtype(v->dtype()), v->shape.size(), &shape[0], (aclFormat)format);
            aclSetTensorFormat(desc, (aclFormat)format);
            aclSetTensorShape(desc, v->shape.size(), &shape[0]);
            LOGv << "aclCreateTensorDesc" << (int)get_dtype(v->dtype()) << v->shape.size() << &shape[0] << format;
            auto data = aclCreateDataBuffer(v->mem_ptr, v->size);
            LOGv << "aclCreateDataBuffer" << v->mem_ptr << v->size;
            ASSERT(desc && data);
            if (is_input) {
                input_desc.push_back(desc);
                input_data.push_back(data);
            } else {
                output_desc.push_back(desc);
                output_data.push_back(data);
            }
        }

        void add_input_host(vector<uint64> v, int dtype=ACL_UINT64) {
            int64_t shape[1];
            shape[0] = v.size();
            auto desc = aclCreateTensorDesc((aclDataType)dtype, 1, &shape[0], ACL_FORMAT_ND);
            aclSetTensorPlaceMent(desc, ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT);
            LOGv << "aclCreateTensorDesc" << dtype << 1 << &shape[0] << ACL_FORMAT_ND;
            auto data = aclCreateDataBuffer(&v[0], v.size()*sizeof(uint64));
            ASSERT(desc && data);
            LOGv << "aclCreateDataBuffer" << &v[0] << v.size()*sizeof(uint64);
            input_desc.push_back(desc);
            input_data.push_back(data);
            input_host.emplace_back(move(v));
            LOGv << "move" << input_host.back().data();
        }

        void add_input_host_scalar(vector<uint64> v, int dtype=ACL_UINT32) {
            int64_t shape[1];
            shape[0] = v.size();
            auto x = (int*)&v[0];
            x[0] = (int32)v[0];
            auto desc = aclCreateTensorDesc((aclDataType)dtype, 0, &shape[0], ACL_FORMAT_ND);
            aclSetTensorPlaceMent(desc, ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT);
            LOGv << "aclCreateTensorDesc" << dtype << 1 << &shape[0] << ACL_FORMAT_ND;
            auto data = aclCreateDataBuffer(&v[0], v.size()*sizeof(uint32));
            ASSERT(desc && data);
            LOGv << "aclCreateDataBuffer" << &v[0] << v.size()*sizeof(uint32);
            input_desc.push_back(desc);
            input_data.push_back(data);
            input_host.emplace_back(move(v));
        }

        void add_input_host_nv(NanoVector nv, int dtype=ACL_UINT64) {
            vector<uint64> v(nv.size());
            for (int i=0; i<nv.size(); i++) v[i] = nv[i];
            int64_t shape[1];
            shape[0] = v.size();
            auto desc = aclCreateTensorDesc((aclDataType)dtype, 1, &shape[0], ACL_FORMAT_ND);
            aclSetTensorPlaceMent(desc, ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT);
            auto data = aclCreateDataBuffer(&v[0], v.size()*sizeof(uint64));
            input_desc.push_back(desc);
            input_data.push_back(data);
            input_host.emplace_back(move(v));
        }

        void add_input_host_nv32(NanoVector nv, int dtype=ACL_INT32) {
            vector<int> v(nv.size());
            for (int i=0; i<nv.size(); i++) v[i] = nv[i];
            int64_t shape[1];
            shape[0] = v.size();
            auto desc = aclCreateTensorDesc((aclDataType)dtype, 1, &shape[0], ACL_FORMAT_ND);
            LOGv << "aclCreateTensorDesc" << dtype << 1 << &shape[0] << ACL_FORMAT_ND << v;
            aclSetTensorPlaceMent(desc, ACL_MEMTYPE_HOST);
            auto data = aclCreateDataBuffer(&v[0], v.size()*sizeof(int));
            input_desc.push_back(desc);
            input_data.push_back(data);
            input_host_32.emplace_back(move(v));
        }
        
        void set_attr(const string& key, bool value) {
            // LOGir << "string bool" << "set_attr" << key << value;
            CHECK(aclopSetAttrBool(attr, key.c_str(), value)==0);
        }
        void set_attr(const string& key, int value, int is_bool) {
            // LOGir << "string bool" << "set_attr" << key << value << is_bool;
            CHECK(aclopSetAttrBool(attr, key.c_str(), value==is_bool)==0);
        }
        void set_attr(const string& key, float value) {
            // LOGir << "string float" <<"set_attr" << key << value;
            CHECK(aclopSetAttrFloat(attr, key.c_str(), value)==0);
        }
        void set_attr(const string& key, int64_t value) {
            // LOGir << "string int64" << "set_attr" << key << value;
            CHECK(aclopSetAttrInt(attr, key.c_str(), value)==0);
        }
        void set_attr(const string& key, int64_t value, int placeholder) {
            // LOGir << "string int64" << "set_attr" << key << value;
            CHECK(aclopSetAttrInt(attr, key.c_str(), value)==0);
        }
        void set_attr(const string& key, int32 value) {
            // LOGir << "string int32" << "set_attr" << key << value;
            CHECK(aclopSetAttrInt(attr, key.c_str(), value)==0);
        }
        void set_attr(const string& key, vector<int64_t> value) {
            // LOGir << "string vector" << "set_attr" << key << value;
            CHECK(aclopSetAttrListInt(attr, key.c_str(), value.size(), &value[0])==0);
        }
        void set_attr(const string& key, string value) {
            // LOGir << "string string" << "set_attr" << key << value;
            CHECK(aclopSetAttrString(attr, key.c_str(), value.c_str())==0);
        }
        void set_attr(const char* key, const char* value) {
            // LOGir << "char" << "set_attr" << key << value;
            CHECK(aclopSetAttrString(attr, key, value)==0);
        }

        void run() {
            // printDeviceData(input_desc, input_data, name);
            
            LOGv << "run" << name << input_desc.size() << output_desc.size();
            if (!PyGILState_Check()) {
                ASSERT(0==aclopCompileAndExecuteV2(name.c_str(), input_desc.size(), &input_desc[0], &input_data[0], output_desc.size(), &output_desc[0], &output_data[0], attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, aclstream));
            } else {
                int ret;
                Py_BEGIN_ALLOW_THREADS
                ret = aclopCompileAndExecuteV2(name.c_str(), input_desc.size(), &input_desc[0], &input_data[0], output_desc.size(), &output_desc[0], &output_data[0], attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, aclstream);
                Py_END_ALLOW_THREADS
                if (ret != 0)
                    LOGf << "aclopCompileAndExecuteV2" << name << "failed return" << ret;
            }
            ASSERT(0==aclrtSynchronizeDevice());

            // printDeviceData(output_desc, output_data, name, false);
        }
    };

    }
    """,
                   cuda_src=f"""
    // aclop
    AclOpRunner op("{name}");
    {input_code}
    {output_code}
    {attr_code}
    op.run();""")


def change_function():
    import jittor as jt
    from jittor import Function

    class IndexACL(Function):

        def __init__(self):
            super(IndexACL, self).__init__()

        def execute(self, inshape: list, dim, dtype="int32"):
            # zeros a tensor, shape is inshape, dtype is dtype
            dim_input = dim
            if dim == None:
                dim = [i for i in range(len(inshape))]
            elif type(dim) == int:
                dim = [dim]
            results = []
            for d in dim:
                max_len = inshape[d]
                tmp = jt.zeros(max_len, dtype=dtype)
                result = acl_cmd(
                    "Range", [jt.Var(0), jt.Var(max_len),
                              jt.Var(1)],
                    output_dtypes=[tmp.dtype],
                    output_shapes=[tmp.shape],
                    attr={})[0]
                broadcast_dim = []
                for i in range(len(inshape)):
                    if i != d:
                        broadcast_dim.append(i)
                result = jt.broadcast(result,
                                      shape=inshape,
                                      dims=broadcast_dim)
                results.append(result)
            if len(results) != 1 or dim_input == None:
                return tuple(results)
            else:
                return results[0]

        def grad(self, grad_output):
            return grad_output

    class PoolACL(Function):

        def get_paddings(self):
            pad_top = self.padding[0]
            pad_left = self.padding[1]
            H = self.input.shape[-2]
            W = self.input.shape[-1]

            totalH = H + 2 * self.padding[0] - self.kernel_size[0]
            totalW = W + 2 * self.padding[1] - self.kernel_size[1]

            kH = (totalH + self.stride[0] -
                  1) // self.stride[0] + 1 if self.attr[
                      'ceil_mode'] else totalH // self.stride[0] + 1
            kW = (totalW + self.stride[1] -
                  1) // self.stride[1] + 1 if self.attr[
                      'ceil_mode'] else totalW // self.stride[1] + 1

            if self.attr['ceil_mode']:
                if (kH - 1) * self.stride[0] >= H + self.padding[0]:
                    kH -= 1
                    need_pad_h = (kH -
                                  1) * self.stride[0] + self.kernel_size[0] - H
                    pad_top = need_pad_h - self.padding[0]
                if (kW - 1) * self.stride[1] >= W + self.padding[1]:
                    kW -= 1
                    need_pad_w = (kW -
                                  1) * self.stride[1] + self.kernel_size[1] - W
                    pad_left = need_pad_w - self.padding[1]

            pads = [self.padding[0], pad_top, self.padding[1], pad_left]
            return pads

        def __init__(self,
                     kernel_size,
                     stride=None,
                     padding=0,
                     dilation=None,
                     return_indices=None,
                     ceil_mode=False,
                     count_include_pad=True,
                     op='maximum'):
            super(PoolACL, self).__init__()
            # set attr
            self.kernel_size = kernel_size if isinstance(
                kernel_size, tuple) else (kernel_size, kernel_size)
            stride = stride if stride else kernel_size
            self.stride = stride if isinstance(stride, tuple) else (stride,
                                                                    stride)
            self.padding = padding if isinstance(padding, tuple) else (padding,
                                                                       padding)
            dilation = dilation if dilation else 1
            self.dilation = dilation if isinstance(
                dilation, tuple) else (dilation, dilation)
            attr = {}

            self.return_indices = return_indices
            self.uint16 = jt.Var(1).int32().dtype
            self.op = op

            if op == 'mean':
                attr['exclusive'] = not count_include_pad
                attr['global_pooling'] = False
                attr['divisor_override_value'] = 0
                attr['ksize'] = [
                    1, 1, self.kernel_size[0], self.kernel_size[1]
                ]
                attr['strides'] = [1, 1, self.stride[0], self.stride[1]]
                attr['ceil_mode'] = ceil_mode
                attr['padding_mode'] = 'CALCULATED'
                attr['data_format'] = 'NCHW'
            elif op == 'maximum':
                attr['ksize'] = [
                    1, self.kernel_size[0], self.kernel_size[1], 1
                ]
                attr['strides'] = [1, self.stride[0], self.stride[1], 1]
                attr['pads'] = [1, self.padding[0], self.padding[1], 1]
                attr['dilation'] = [1, self.dilation[0], self.dilation[1], 1]
                # attr['ceil_mode'] = ceil_mode

            self.attr = attr

        def execute(self, input):

            # create input
            input_shape = input.shape
            input_dtype = input.dtype

            self.input = input
            # create output
            output_shape = [
                input_shape[0], input_shape[1],
                (input_shape[2] + 2 * self.padding[0] - self.dilation[0] *
                 (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1,
                (input_shape[3] + 2 * self.padding[1] - self.dilation[1] *
                 (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
            ]
            output_dtype = input_dtype

            if self.op == 'mean':
                self.attr['pads'] = self.get_paddings()
                result = acl_cmd("AvgPoolV2", [input],
                                 output_dtypes=[output_dtype],
                                 output_shapes=[output_shape],
                                 attr=self.attr)
            elif self.op == 'maximum':
                result = acl_cmd("MaxPoolWithArgmaxV1", [input],
                                 output_dtypes=[output_dtype, self.uint16],
                                 output_shapes=[output_shape, output_shape],
                                 attr=self.attr)
            else:
                raise ValueError('no this type pool')

            if self.op == 'maximum':
                self.index = result[1]

            if self.return_indices:
                return result[0], result[1]
            else:
                return result[0]

        def grad(self, grad_output):
            if self.op == 'maximum':
                grad_input = acl_cmd("MaxPoolGradWithArgmaxV1",
                                     [self.input, grad_output, self.index],
                                     output_dtypes=[grad_output.dtype],
                                     output_shapes=[self.input.shape],
                                     attr=self.attr)[0]
            elif self.op == 'mean':
                grad_input = acl_cmd("AvgPoolV2",
                                     [self.input, grad_output, self.index],
                                     output_dtypes=[grad_output.dtype],
                                     output_shapes=[self.input.shape],
                                     attr=self.attr)[0]
            else:
                grad_input = None
            return grad_input

    class BmmACL(Function):

        def __init__(self, adj_x1=False, adj_x2=False):
            super(BmmACL, self).__init__()
            self.adj_x1 = adj_x1
            self.adj_x2 = adj_x2

        def execute(self, x1, x2):
            self.input = [x1, x2]
            result = acl_cmd("BatchMatMul", [x1, x2],
                             output_dtypes=[x1.dtype],
                             output_shapes=[x1.shape[:-1] + x2.shape[-1:]],
                             attr={})[0]
            return result

        def grad(self, grad_output):
            x1, x2 = self.input
            grad_x1 = acl_cmd(
                "BatchMatMul", [grad_output, x2.transpose(-2, -1)],
                output_dtypes=[x1.dtype],
                output_shapes=[grad_output.shape[:-1] + x1.shape[-1:]],
                attr={})[0]
            grad_x2 = acl_cmd(
                "BatchMatMul", [x1.transpose(-2, -1), grad_output],
                output_dtypes=[x2.dtype],
                output_shapes=[x2.shape[:-1] + grad_output.shape[-1:]],
                attr={})[0]
            return grad_x1, grad_x2

    class MatmulACL(Function):

        def __init__(self, adj_x1=False, adj_x2=False):
            super(MatmulACL, self).__init__()
            self.adj_x1 = adj_x1
            self.adj_x2 = adj_x2

        def execute(self, x1, x2):
            self.input = [x1, x2]
            if len(x1.shape) > 2 or len(x2.shape) > 2:
                result = acl_cmd("BatchMatMul", [x1, x2],
                                 output_dtypes=[x1.dtype],
                                 output_shapes=[x1.shape[:-1] + x2.shape[-1:]],
                                 attr={})[0]
            else:
                result = acl_cmd("MatMul", [x1, x2],
                                 output_dtypes=[x1.dtype],
                                 output_shapes=[x1.shape[:-1] + x2.shape[-1:]],
                                 attr={})[0]
            return result

        def grad(self, grad_output):
            x1, x2 = self.input
            if len(x1.shape) > 2 or len(x2.shape) > 2:
                grad_x1 = acl_cmd(
                    "BatchMatMul",
                    [grad_output, x2.transpose(-2, -1)],
                    output_dtypes=[x1.dtype],
                    output_shapes=[grad_output.shape[:-1] + x1.shape[-1:]],
                    attr={})[0]
                grad_x2 = acl_cmd(
                    "BatchMatMul", [x1.transpose(-2, -1), grad_output],
                    output_dtypes=[x2.dtype],
                    output_shapes=[x2.shape[:-1] + grad_output.shape[-1:]],
                    attr={})[0]
            else:
                grad_x1 = acl_cmd(
                    "MatMul", [grad_output, x2.transpose(-2, -1)],
                    output_dtypes=[x1.dtype],
                    output_shapes=[grad_output.shape[:-1] + x1.shape[-1:]],
                    attr={})[0]
                grad_x2 = acl_cmd(
                    "MatMul", [x1.transpose(-2, -1), grad_output],
                    output_dtypes=[x2.dtype],
                    output_shapes=[x2.shape[:-1] + grad_output.shape[-1:]],
                    attr={})[0]
            return grad_x1, grad_x2

    class GetItem(Function):

        def __init__(self):
            super(GetItem, self).__init__()
            self.type_ = 'index'

        def stride(self, x, dim):
            stride = 1
            for i in range(dim + 1, len(x.shape)):
                stride *= x.shape[i]
            return stride

        def execute(self, x, slices, return_x=None):
            if isinstance(slices, jt.Var) or isinstance(slices, tuple):
                if isinstance(slices, jt.Var):
                    slices = (slices, )
                if isinstance(slices[0], jt.Var):
                    slices_len = len(slices)
                    masks = jt.ones(slices_len, dtype=jt.int64)
                    output = slices[0].shape
                    output += x.shape[slices_len:]
                    input_ = [x, masks, jt.Var(list(output)).int64()]
                    for i in range(slices_len):
                        input_.append(slices[i].int32())
                    result = acl_cmd("Index",
                                     input_,
                                     output_dtypes=[x.dtype],
                                     output_shapes=[output],
                                     attr={})[0]
                    self.shape = x.shape
                    self.sizes = list(output)
                    self.type_ = 'index'
                    self.slices = slices
                    # self.strides
                    return result

            # use AsStrided operator to implement the getitem function
            # get the shape and stride of the input tensor
            x_dim = len(x.shape)
            # int type
            if not isinstance(slices, tuple):
                slices = (slices, )

            if len(slices) < x_dim:
                slices += (slice(None, None, None), ) * (x_dim - len(slices))

            self.inputs = [x, slices]

            sizes = []
            strides = []
            offset = 0

            for dim, s in enumerate(slices):
                if isinstance(s, int):
                    if s < 0:  # Handle negative indices.
                        s += x.shape[dim]
                    offset += s * self.stride(x, dim)
                elif isinstance(s, slice):
                    # Unpack the slice
                    start, stop, step = s.indices(x.size(dim))
                    size = (stop - start - 1) // step + 1
                    stride = self.stride(x, dim) * step
                    offset += start * self.stride(x, dim)
                    sizes.append(size)
                    strides.append(stride)
                else:
                    raise ValueError("Invalid slice type")

            if not sizes:
                sizes = [1]
                strides = [0]
            # AsStrided same with as_strided of pytorch
            self.sizes = sizes
            self.strides = strides
            self.offset = offset
            self.shape = x.shape
            self.type_ = 'as_strided'
            result = acl_cmd(
                "AsStrided",
                [x, jt.Var(sizes),
                 jt.Var(strides),
                 jt.Var(offset)],
                output_dtypes=[x.dtype],
                output_shapes=[jt.empty(sizes).shape],
                attr={})[0]
            return result

        def grad(self, grad_output):
            if self.type_ == 'as_strided':
                result = jt.zeros(self.shape, dtype=grad_output.dtype)
                sizes = list(grad_output.shape)
                strides = [
                    self.stride(grad_output, dim)
                    for dim in range(len(grad_output.shape))
                ]
                result = acl_cmd("ViewCopy", [
                    result,
                    jt.Var(self.sizes),
                    jt.Var(self.strides),
                    jt.Var(self.offset), grad_output,
                    jt.Var(sizes),
                    jt.Var(strides),
                    jt.Var(0)
                ],
                                 output_dtypes=[result.dtype],
                                 output_shapes=[result.shape],
                                 attr={})[0]
            elif self.type_ == 'index':
                #TODO: use IndexPutV2 to implement the grad function
                assert len(self.slices) == 1
                index = self.slices[0]
                input = jt.zeros(self.shape, dtype=grad_output.dtype)
                input_flatten = input.reshape(input.shape[0], -1)
                index_flatten = index.reshape(-1).unsqueeze(-1).repeat(
                    1, input_flatten.shape[1])
                grad_output_flatten = grad_output.reshape(index.numel(), -1)
                result = acl_cmd(
                    "ScatterElements",
                    [input_flatten, index_flatten, grad_output_flatten],
                    output_dtypes=[input.dtype],
                    output_shapes=[input.shape],
                    attr={
                        'axis': 0,
                        'reduction': 'add'
                    })[0]
                result = result.reshape(self.shape)
                # result = jt.zeros(self.shape, dtype=grad_output.dtype)
                # # masks = jt.ones(len(self.slices), dtype=jt.int64)
                # masks = jt.array([1,1], dtype=jt.int64)
                # expand_masks = jt.array([1,1], dtype=jt.int64)
                # inputs_ = [result,grad_output,masks,expand_masks]
                # slices_len = len(self.slices)
                # for  i in range(slices_len):
                #     inputs_.append(self.slices[i].int64())
                # # breakpoint()
                # jt.sync_all(True)
                # print(inputs_)
                # result_ = acl_cmd("IndexPutV2", inputs_,
                #                  output_dtypes=[result.dtype],
                #                  output_shapes=[result.shape],
                #                  attr={"accumulate":True})[0]
                # result = result_
            else:
                raise ValueError("Invalid slice type")
            result.sync()
            return result, None

    class ConcatACL(Function):

        def __init__(self):
            super(ConcatACL, self).__init__()

        def execute(self, input_tensors, dim=0):
            self.input = input_tensors
            for i in range(len(input_tensors)):
                if input_tensors[i].dtype != input_tensors[0].dtype:
                    raise ValueError(
                        "All input tensors must have the same dtype")
                if input_tensors[i].shape[:dim] != input_tensors[
                        0].shape[:dim] or input_tensors[i].shape[
                            dim + 1:] != input_tensors[0].shape[dim + 1:]:
                    raise ValueError(
                        "All input tensors must have the same shape")
            result = acl_cmd(
                "ConcatD",
                input_tensors,
                output_dtypes=[input_tensors[0].dtype],
                output_shapes=[
                    jt.empty(self.calculate_output_shape(input_tensors,
                                                         dim)).shape
                ],
                attr={
                    "N": len(input_tensors),
                    "concat_dim": dim
                })[0]
            return result

        def grad(self, grad_output):
            grad_inputs = self.split_grad(grad_output, self.input, self.axis)
            return grad_inputs

        def calculate_output_shape(self, input_tensors, axis):
            shape = list(input_tensors[0].shape)
            for tensor in input_tensors[1:]:
                shape[axis] += tensor.shape[axis]
            return tuple(shape)

        def split_grad(self, grad_output, input_tensors, axis):
            offset = 0
            grad_inputs = []
            for tensor in input_tensors:
                grad_input = acl_cmd("Slice", [
                    grad_output, [0] * axis + [offset] + [0] *
                    (len(tensor.shape) - axis - 1), tensor.shape
                ])
                grad_inputs.append(grad_input)
                offset += tensor.shape[axis]
            return grad_inputs

    class SetItemACL(Function):

        def __init__(self):
            super(SetItemACL, self).__init__()

        def stride(self, x, dim):
            # 计算给定维度的步长
            stride = 1
            for i in range(dim + 1, len(x.shape)):
                stride *= x.shape[i]
            return stride

        def execute(self, x, slices, value, reduce='void'):
            self.is_tensor = type(value) == jt.Var
            if type(value) != jt.Var:
                value = jt.array(value)
            x_dim = len(x.shape)

            # 确保slices是一个元组
            if not isinstance(slices, tuple):
                slices = (slices, )

            # 补齐slices使其长度等于x的维度
            if len(slices) < x_dim:
                slices += (slice(None, None, None), ) * (x_dim - len(slices))

            self.inputs = [x, slices, value]

            target_sizes = []
            target_strides = []
            offset = 0

            for dim, s in enumerate(slices):
                if isinstance(s, int):
                    if s < 0:
                        s += x.shape[dim]
                    s = slice(s, s + 1, None)
                if isinstance(s, slice):
                    # 解包切片
                    start, stop, step = s.indices(x.shape[dim])
                    size = (stop - start - 1) // step + 1
                    stride = self.stride(x, dim) * step
                    offset += start * self.stride(x, dim)
                    target_sizes.append(size)
                    target_strides.append(stride)
                else:
                    print("slices: ", s, type(s))
                    raise ValueError("Invalid slice type")

            # 计算value的size、stride和offset
            value_sizes = list(value.shape)
            value_strides = [
                self.stride(value, dim) for dim in range(len(value.shape))
            ]

            self.target_sizes = target_sizes
            self.target_strides = target_strides
            self.offset = offset
            self.value_sizes = value_sizes
            self.value_strides = value_strides

            #import pdb; pdb.set_trace()
            result = acl_cmd("ViewCopy", [
                x,
                jt.Var(target_sizes),
                jt.Var(target_strides),
                jt.Var(offset), value,
                jt.Var(value_sizes),
                jt.Var(value_strides),
                jt.Var(0)
            ],
                             output_dtypes=[x.dtype],
                             output_shapes=[x.shape],
                             attr={})[0]
            result.sync()
            return result

        def grad(self, grad_output):
            result = acl_cmd("AsStrided", [
                grad_output,
                jt.Var(self.target_sizes),
                jt.Var(self.target_strides),
                jt.Var(self.offset)
            ],
                             output_dtypes=[grad_output.dtype],
                             output_shapes=[jt.empty(self.target_sizes).shape],
                             attr={})[0]
            # copy grad_output to new_grad_output
            new_grad_output = acl_cmd("Copy", [grad_output],
                                      output_dtypes=[grad_output.dtype],
                                      output_shapes=[grad_output.shape],
                                      attr={"N": 1})[0]
            new_grad_output = acl_cmd("ViewCopy", [
                new_grad_output,
                jt.Var(self.target_sizes),
                jt.Var(self.target_strides),
                jt.Var(self.offset),
                jt.zeros(self.value_sizes, dtype=grad_output.dtype),
                jt.Var(self.value_sizes),
                jt.Var(self.value_strides),
                jt.Var(0)
            ],
                                      output_dtypes=[grad_output.dtype],
                                      output_shapes=[grad_output.shape],
                                      attr={})[0]
            new_grad_output.sync()
            return new_grad_output, None, result if self.is_tensor else None

    class TriuACL(Function):

        def __init__(self):
            super(TriuACL, self).__init__()

        def execute(self, input, k):
            self.input = input
            result = acl_cmd("Triu", [input],
                             output_dtypes=[input.dtype],
                             output_shapes=[input.shape],
                             attr={'diagonal': k})[0]
            return result

        def grad(self, grad_output):
            return grad_output

    class TransposeACL(Function):

        def __init__(self):
            super(TransposeACL, self).__init__()

        def execute(self, input, perm):
            self.input = input

            output_shape = input.shape[perm[0]:perm[0] + 1]

            for i in range(1, len(perm)):
                output_shape += input.shape[perm[i]:perm[i] + 1]
            result = acl_cmd("Transpose", [input, jt.Var(perm)],
                             output_dtypes=[input.dtype],
                             output_shapes=[output_shape],
                             attr={})[0]
            return result

        def grad(self, grad_output):
            return grad_output

    class AdaptiveMaxPool2dACL(Function):

        def __init__(
            self,
            output_size,
            return_indices=False,
        ):
            super(AdaptiveMaxPool2dACL, self).__init__()
            self.output_size = (output_size, output_size) if isinstance(
                output_size, int) else output_size

            self.return_indices = return_indices
            self.uint16 = jt.Var(1).int32().dtype

            attr = {}
            attr['ceil_mode'] = False
            attr['dilations'] = [1, 1, 1, 1]
            self.attr = attr

        def execute(self, input):
            input_shape = input.shape
            input_dtype = input.dtype

            output_shape = [
                input_shape[0], input_shape[1], self.output_size[0],
                self.output_size[1]
            ]
            output_dtype = input_dtype
            self.input = input

            stride_h = input_shape[2] // output_shape[2]
            stride_w = input_shape[3] // output_shape[3]
            kernel_size_h = input_shape[2] - (output_shape[2] - 1) * stride_h
            kernel_size_w = input_shape[3] - (output_shape[3] - 1) * stride_w

            stride = [0, 0]
            kernel_size = [0, 0]
            padding = [0, 0]

            stride[0] = stride_h
            stride[1] = stride_w
            kernel_size[0] = kernel_size_h
            kernel_size[1] = kernel_size_w
            padding[0] = padding[1] = 0
            kernel_sizes = [1, kernel_size[0], kernel_size[1], 1]
            strides_size = [1, stride[0], stride[1], 1]
            paddings = [1, padding[0], padding[1], 1]

            self.attr['ksize'] = kernel_sizes
            self.attr['strides'] = strides_size
            self.attr['pads'] = paddings

            result = acl_cmd("MaxPoolWithArgmaxV1", [input],
                             output_dtypes=[output_dtype, self.uint16],
                             output_shapes=[output_shape, output_shape],
                             attr=self.attr)

            self.index = result[1]

            if self.return_indices:
                return result[0], result[1]
            else:
                return result[0]

        def grad(self, grad_output):
            grad_input = acl_cmd("MaxPoolGradWithArgmaxV1",
                                 [self.input, grad_output, self.index],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[self.input.shape],
                                 attr=self.attr)[0]
            return grad_input

    class AdaptiveAvgPool2dACL(Function):

        def __init__(self, output_size):
            super(AdaptiveAvgPool2dACL, self).__init__()
            self.output_size = (output_size, output_size) if isinstance(
                output_size, int) else output_size

            attr = {}
            if isinstance(output_size, tuple):
                output_size = [output_size[0], output_size[1]]
            attr['output_size'] = output_size
            self.attr = attr

        def execute(self, input):
            input_shape = input.shape
            input_dtype = input.dtype

            self.original_shape = input_shape

            output_shape = [
                input_shape[0], input_shape[1], self.attr['output_size'][0],
                self.attr['output_size'][1]
            ]
            output_dtype = input_dtype
            self.input = input

            result = acl_cmd("AdaptiveAvgPool2d", [input],
                             output_dtypes=[output_dtype],
                             output_shapes=[output_shape],
                             attr=self.attr)

            return result[0]

        def grad(self, grad_output):
            attr = {}
            attr['orig_input_shape'] = list(self.original_shape)
            grad_input = acl_cmd("AdaptiveAvgPool2dGrad", [grad_output],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[self.original_shape],
                                 attr=attr)[0]
            return grad_input

    class CumsumACL(Function):

        def __init__(self):
            super(CumsumACL, self).__init__()

        def execute(self, input, dim=-1):
            self.input = input
            self.dim = dim
            result = acl_cmd("Cumsum", [input, jt.Var(dim)],
                             output_dtypes=[input.dtype],
                             output_shapes=[input.shape],
                             attr={})[0]
            return result

        def grad(self, grad_output):
            flipped_grad_output = acl_cmd(
                "ReverseV2", [grad_output, jt.Var([self.dim])],
                output_dtypes=[grad_output.dtype],
                output_shapes=[grad_output.shape],
                attr={})[0]
            cumulative_grad = acl_cmd(
                "Cumsum",
                [flipped_grad_output, jt.Var(self.dim)],
                output_dtypes=[grad_output.dtype],
                output_shapes=[grad_output.shape],
                attr={})[0]
            grad_input = acl_cmd(
                "ReverseV2",
                [cumulative_grad, jt.Var([self.dim])],
                output_dtypes=[grad_output.dtype],
                output_shapes=[grad_output.shape],
                attr={})[0]
            return grad_input

    class GatherACL(Function):

        def __init__(self):
            super(GatherACL, self).__init__()

        def execute(self, input, dim, index):
            self.input = input
            self.dim = dim
            self.index = index

            result = acl_cmd("GatherElements", [input, index],
                             output_dtypes=[input.dtype],
                             output_shapes=[index.shape],
                             attr={'dim': dim})[0]
            return result

        def grad(self, grad_output):
            tmp = jt.zeros(self.index.shape, dtype=grad_output.dtype)
            grad_input = acl_cmd("ScatterElements",
                                 [tmp, self.index, grad_output],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[tmp.shape],
                                 attr={
                                     'axis': self.dim,
                                     'reduction': "add"
                                 })[0]
            return grad_input

    class ScatterACL(Function):

        def __init__(self):
            super(ScatterACL, self).__init__()

        def execute(self, input, dim, index, src, reduce='void'):
            self.input = input
            self.dim = dim
            self.index = index
            self.reduce = reduce
            result = acl_cmd("ScatterElements", [input, self.index, src],
                             output_dtypes=[input.dtype],
                             output_shapes=[input.shape],
                             attr={
                                 'axis': self.dim,
                                 'reduction': reduce
                             })[0]
            return result

        def grad(self, grad_output):
            grad_input = acl_cmd("GatherElements", [grad_output, self.index],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[self.index.shape],
                                 attr={'dim': self.dim})[0]
            return grad_output, None, None, grad_input

    class WhereACL(Function):

        def __init__(self):
            super(WhereACL, self).__init__()

        def execute(self, condition, x, y):
            self.condition = condition

            if x.dtype != y.dtype:
                if x.dtype == jt.float32:
                    y = y.float32()
                elif y.dtype == jt.float32:
                    x = x.float32()
                else:
                    x = x.to(y.dtype)

            self.x = x
            self.y = y

            result = acl_cmd("Select", [condition, x, y],
                             output_dtypes=[x.dtype],
                             output_shapes=[x.shape],
                             attr={})[0]
            return result

        def grad(self, grad_output):
            tmp = jt.zeros(grad_output.shape, dtype=grad_output.dtype)
            grad_x = acl_cmd("Select", [self.condition, grad_output, tmp],
                             output_dtypes=[self.x.dtype],
                             output_shapes=[self.x.shape],
                             attr={})[0]

            grad_y = acl_cmd("Select", [self.condition, tmp, grad_output],
                             output_dtypes=[self.y.dtype],
                             output_shapes=[self.y.shape],
                             attr={})[0]
            return grad_output, grad_x, grad_y

    class FlipACL(Function):

        def __init__(self):
            super(FlipACL, self).__init__()

        def execute(self, input, dim):
            self.input = input
            #if isinstance(dim_vector, tuple):
            dim_vector = jt.Var(list(dim))
            #print(dim_vector.dtype)
            self.dim_vector = dim_vector
            #print(input, dim_vector)
            result = acl_cmd("ReverseV2", [input, dim_vector],
                             output_dtypes=[input.dtype],
                             output_shapes=[input.shape],
                             attr={})[0]
            return result

        def grad(self, grad_output):
            #print(grad_output)
            grad_input = acl_cmd("ReverseV2", [grad_output, self.dim_vector],
                                 output_dtypes=[grad_output.dtype],
                                 output_shapes=[grad_output.shape],
                                 attr={})[0]
            return grad_input

    class FloorIntACL(Function):

        def __init__(self):
            super(FloorIntACL, self).__init__()

        def execute(self, input):
            self.input = input
            self.shape = input.shape
            result = acl_cmd("Floor", [input],
                             output_dtypes=[jt.int],
                             output_shapes=[input.shape],
                             attr={})[0]
            return result

        def grad(self, grad_output):
            return jt.zeros(self.shape, dtype=grad_output.dtype)

    def warp(origin_func, new_func):

        def warpper(*args, **kwargs):
            if origin_func == jt.index:
                if len(args) == 2 and args[1] == None:
                    args = tuple(list(args[0:1]))
            if jt.flags.use_acl:
                if isinstance(new_func, IndexACL):
                    if len(args) == 1:
                        args = (args[0], None)
                if isinstance(new_func, CumsumACL):
                    args = (args[0], kwargs.get('dim', -1))
                    kwargs = {}
                if isinstance(new_func,
                              ScatterACL) and kwargs.get('reduce') is not None:
                    args = (args[0], args[1], args[2], args[3],
                            kwargs.get('reduce', 'void'))
                    kwargs = {}

                return new_func(*args, **kwargs)
            return origin_func(*args, **kwargs)

        return warpper

    jt.index = warp(jt.index, IndexACL())
    jt.Var.index = lambda x, dim=None: warp(jt.index, IndexACL())(x.shape, dim)
    jt.nn.Pool = warp(jt.nn.Pool, PoolACL)
    jt.nn.AdaptiveMaxPool2d = warp(jt.nn.AdaptiveMaxPool2d,
                                   AdaptiveMaxPool2dACL)
    jt.nn.AdaptiveAvgPool2d = warp(jt.nn.AdaptiveAvgPool2d,
                                   AdaptiveAvgPool2dACL)

    jt.triu = warp(jt.triu, TriuACL())
    jt.triu_ = warp(jt.triu, TriuACL())
    jt.Var.triu = lambda x: warp(jt.Var.triu, TriuACL())(x)
    jt.Var.triu_ = lambda x: warp(jt.Var.triu_, TriuACL())(x)

    jt.getitem = warp(jt.getitem, GetItem())
    jt.Var.getitem = lambda x, slices, return_x=None: warp(
        jt.getitem, GetItem())(x, slices)

    jt.setitem = warp(jt.setitem, SetItemACL())
    jt.Var.setitem = lambda x, slices, value, reduce='void': warp(
        jt.setitem, SetItemACL())(x, slices, value, reduce)

    jt.misc.flip = warp(jt.misc.flip, FlipACL())
    jt.Var.flip = lambda x, dim_vector: warp(jt.misc.flip, FlipACL())(
        x, dim_vector)
    jt.cumsum = warp(jt.cumsum, CumsumACL())
    jt.gather = warp(jt.gather, GatherACL())
    jt.Var.gather = lambda x, dim, index: warp(jt.gather, GatherACL())(x, dim,
                                                                       index)
    jt.scatter = warp(jt.scatter, ScatterACL())
    jt.Var.scatter = lambda x, dim, index, src, reduce="void": warp(
        jt.scatter, ScatterACL())(x, dim, index, src, reduce)
    jt.where = warp(jt.where, WhereACL())
    jt.floor_int = warp(jt.floor_int, FloorIntACL())
    jt.Var.floor_int = lambda x: warp(jt.floor_int, FloorIntACL())(x)

    # jt.nn.bmm = warp(jt.nn.bmm, BmmACL())
    # jt.bmm = warp(jt.bmm, BmmACL())
    # jt.nn.matmul = warp(jt.matmul, MatmulACL())
    # jt.matmul = warp(jt.matmul, MatmulACL())
    # jt.transpose = warp(jt.transpose, TransposeACL())
    # jt.Var.transpose = lambda x, perm: warp(jt.transpose, TransposeACL())(x, perm)
    # jt.concat = warp(jt.concat, ConcatACL())
