// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
#include "common.h"
#include "op.h"
#include "acl_jittor.h"
#include "ops/random_op.h"
#include "ops/reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/transpose_op.h"
#include "ops/array_op.h"
#include "ops/code_op.h"
#include "fused_op.h"
#include "ops/unary_op.h"
#include "ops/ternary_op.h"
#include "executor.h"
#include "misc/cuda_flags.h"
#include "mem/allocator.h"
#include "op_compiler.h"
#include "ops/op_register.h"
#include "opt/tuner_manager.h"
#include "utils/str_utils.h"

namespace jittor {

using std::swap;

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
        //     if(i >= 3)
        //         break;
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

void free_var_mem(Var* v);


unordered_map<uint32, string> opname_map = {
    // unary op
    {ns_cast, "Cast"},
    {ns_negative, "Neg"},
    {ns_abs, "Abs"},
    {ns_exp, "Exp"},
    {ns_log, "Log"},
    {ns_sqrt, "Sqrt"},
    {ns_ceil, "Ceil"},
    {ns_floor, "Floor"},
    {ns_round, "Round"},
    // m(round_int)
    // m(floor_int)
    // m(ceil_int)
    {ns_sin, "Sin"},
    {ns_cos, "Cos"},
    {ns_tan, "Tan"},
    {ns_asin, "Asin"},
    {ns_acos, "Acos"},
    {ns_atan, "Atan"},
    {ns_sinh, "Sinh"},
    {ns_cosh, "Cosh"},
    {ns_tanh, "Tanh"},
    {ns_asinh, "Asinh"},
    {ns_acosh, "Acosh"},
    {ns_atanh, "Atanh"},
    {ns_sigmoid, "Sigmoid"},
    {ns_erf, "Erf"},
    {ns_erfinv, "Erfinv"},
    {ns_logical_not, "LogicalNot"},
    {ns_bitwise_not, "BitwiseNot"},
    // binary op
    {ns_pow, "Pow"},
    {ns_maximum, "Maximum"},
    {ns_minimum, "Minimum"},
    {ns_add, "Add"},
    {ns_subtract, "Sub"},
    {ns_multiply, "Mul"},
    {ns_divide, "RealDiv"},
    {ns_floor_divide, "FloorDiv"},
    {ns_mod, "Mod"},
    {ns_less, "Less"},
    {ns_less_equal, "LessEqual"},
    {ns_greater, "Greater"},
    {ns_greater_equal, "GreaterEqual"},
    {ns_equal, "Equal"},
    {ns_not_equal, "NotEqual"},
    {ns_left_shift, "LeftShift"},
    {ns_right_shift, "RightShift"},
    {ns_logical_and, "LogicalAnd"},
    {ns_logical_or, "LogicalOr"},
    {ns_logical_xor, "LogicalXor"},
    {ns_bitwise_and, "BitwiseAnd"},
    {ns_bitwise_or, "BitwiseOr"},
    {ns_bitwise_xor, "BitwiseXor"},

};

void fallback_cpu(Op* op) {
    LOGy << "!!! fallback_cpu " << op;
    use_cuda = 0;
    for (auto v : op->inputs()) {
        if (v->mem_ptr && v->allocator->is_cuda()) {
            migrate_to_cpu(v, exe.allocator);
        }
    }
    for (auto v : op->outputs()) {
        if (v->mem_ptr && v->allocator->is_cuda()) {
            migrate_to_cpu(v, exe.allocator);
        }
    }
    op->flags.set(NodeFlags::_cpu);
    op->flags.set(NodeFlags::_cuda, 0);
    if (op->name() == string("fused")) {
        auto fop = (FusedOp*)op;
        for (auto op : fop->ops) {
            op->flags.set(NodeFlags::_cpu);
            op->flags.set(NodeFlags::_cuda, 0);
        }
    }
    op->do_run();
    use_cuda = 1;
}

/*
    check compile
    if compiled: exec
    else: compile
        check is fused
            check is relay
            else
                compile func = try exec
                    if failed: fallback_cpu
        else
            try compile
            if failed: fallback_cpu
*/

extern jit_op_entry_t (*do_compile_hook)(Op*);
jit_op_entry_t do_compile_inner(Op* op);

void try_exec_and_fallback_cpu(Op* op) {
    LOGv << "try_exec_and_fallback_cpu " << op;
    auto fop = (FusedOp*)op;

    vector<Var*> new_alloced;
    int fallback = 0;
    try {
        for (Op* op : fop->ops) {
            for (auto out : op->outputs()) {
                if (out->mem_ptr) continue;
                out->alloc(exe.temp_allocator);
                new_alloced.push_back(out);
            }
            if (op->name() == string("unary")) {
                auto uop = (UnaryOp*)op;
                AclOpRunner op("...");
                op.add(uop->x, true);
                op.add(uop->y, false);
                if (uop->ns == ns_cast) {
                    op.set_attr("dst_type", (int64_t)op.get_dtype(uop->y->dtype()));
                }
                auto iter = opname_map.find(uop->ns);
                ASSERT(iter != opname_map.end()) << "op " << uop->ns << " not found";
                op.name = iter->second;
                op.run();
            } else
            if (op->name() == string("binary")) {
                auto bop = (BinaryOp*)op;
                AclOpRunner op("...");
                op.add(bop->x, true);
                op.add(bop->y, true);
                op.add(bop->z, false);
                auto iter = opname_map.find(bop->ns);
                ASSERT(iter != opname_map.end()) << "op " << bop->ns << " not found";
                op.name = iter->second;
                if (bop->x->dtype() == ns_bool and bop->y->dtype() == ns_bool)
                {
                    // BitwiseOr, BitwiseAnd, BitwiseXor -> LogicalOr, LogicalAnd, LogicalXor
                    if (bop->ns == ns_bitwise_or) {
                        op.name = "LogicalOr";
                    } else if (bop->ns == ns_bitwise_and) {
                        op.name = "LogicalAnd";
                    } else if (bop->ns == ns_bitwise_xor) {
                        op.name = "LogicalXor";
                    }
                }
                op.run();
            } else
            if (op->name() == string("ternary")) {
                auto top = (TernaryOp*)op;
                AclOpRunner op("Select");
                op.add(top->cond, true);
                op.add(top->x, true);
                op.add(top->y, true);
                op.add(top->z, false);
                op.run();
            } else 
            if (op->name() == string("array")) {
                auto aop = (ArrayOp*)op;
                aclrtMemcpy(aop->output->mem_ptr, aop->output->size, aop->ptr<void>(), aop->output->size, ACL_MEMCPY_HOST_TO_DEVICE);         
            } else 
            if (op->name() == string("reduce")) {
                auto rop = (ReduceOp*)op;
                AclOpRunner op("");
                if (rop->ns == ns_add)
                    op.name = "ReduceSum";
                else if (rop->ns == ns_multiply)
                    op.name = "ReduceProd";
                else if (rop->ns == ns_maximum)
                    op.name = "ReduceMax";
                else if (rop->ns == ns_minimum)
                    op.name = "ReduceMin";
                else if (rop->ns == ns_mean)
                    op.name = "ReduceMean";
                else
                    LOGf << "op " << rop->ns << " not supported";
                op.add(rop->x, true);
                vector<uint64> axes;
                for (int i=0; i<rop->x->shape.size(); i++)
                    if (rop->reduce_mask & (1<<i))
                        axes.push_back(i);
                // op.set_attr("axes", axes);
                op.add_input_host(axes, ACL_INT64);
                op.add(rop->y, false);
                op.set_attr("keep_dims", false);
                if (rop->ns == ns_mean) {
                    // operation: An optional int32 from 1(SUM), 2(ASUM), 3(SUMSQ), and 4(MEAN), specifying the reduction algorithm. Defaults to "1". 
                    op.set_attr("operation", 4);
                }
                op.run();
            } else
            if (op->name() == string("broadcast_to")) {
                auto bop = (BroadcastToOp*)op;
                AclOpRunner op("Expand");
                NanoVector xshape, xshape_bk = bop->x->shape;
                NanoVector zshape = bop->z->shape;
                for (int i=0; i<zshape.size(); i++) {
                    if (bop->bcast_mask & (1<<i)) {
                        xshape.push_back(1);
                    } else {
                        xshape.push_back(zshape[i]);
                    }
                }
                bop->x->shape = xshape;
                op.add(bop->x, true);
                bop->x->shape = xshape_bk;
                op.add_input_host_nv(zshape, ACL_INT64);
                op.add(bop->z, false);
                op.run();
            } 
            else
            if (op->name() == string("fuse_transpose")) {
                // replace fuse_transpose with transpose
                auto top = (TransposeOp*)op;
                AclOpRunner op("Transpose");
                op.add(top->x, true);
                op.add(top->y, false);
                vector<uint64> axes;
                for (int i=0; i<top->axes.size(); i++)
                    axes.push_back(top->axes[i]);
                op.add_input_host(axes, ACL_INT64);
                op.run();
            } else
            {
                LOGf << "op " << op->name() << " not supported";
            }
        }
    } catch (std::exception& e) {
        fallback = 1;
        LOGir << "fallback cpu" << e.what();
    }
    for (auto v : new_alloced) {
        free_var_mem(v);
    }
    if (fallback) {
        fallback_cpu(op);
    }
}

extern int current_seed;
extern int64 current_offset;

static unordered_map<string, std::function<void(Op*)>> acl_ops = {
{"curand_random", [&current_seed, &current_offset](Op* op) {
    auto _op = (RandomOp*)op; 
    AclOpRunner runner(_op->type == ns_uniform ? "StatelessRandomUniformV2" : "StatelessRandomNormalV2");
    auto out = op->output(0);
    runner.add_input_host_nv(out->shape, ACL_INT64); // shape
    runner.add_input_host({current_seed}); // seed
    runner.add_input_host({0,current_offset}); // offset
    runner.add_input_host_scalar({1}, ACL_INT32); // algorithm
    runner.add(out, false);
    runner.set_attr("dtype", (int64_t)runner.get_dtype(out->dtype()));
    runner.run();
    // aclrtSynchronizeDevice();
    current_offset += out->numel();
}},
{"cublas_matmul", [&](Op* op) {
    struct MatmulOp : Op {
        Var* a, *b, *c;
        bool trans_a, trans_b;
    };
    auto _op = (MatmulOp*)op;
    AclOpRunner runner("MatMul");
    runner.add(_op->a, true);
    runner.add(_op->b, true);
    runner.add(_op->c, false);
    runner.set_attr("transpose_x1", _op->trans_a);
    runner.set_attr("transpose_x2", _op->trans_b);
    runner.run();
}},
{"cublas_batched_matmul", [&](Op* op) {
    struct BatchedMatmulOp : Op {
        Var* a, *b, *c;
        bool adj_x1, adj_x2;
    };
    auto _op = (BatchedMatmulOp*)op;
    AclOpRunner runner("BatchMatMul");
    runner.add(_op->a, true);
    runner.add(_op->b, true);
    runner.add(_op->c, false);
    runner.set_attr("adj_x1", _op->adj_x1);
    runner.set_attr("adj_x2", _op->adj_x2);
    runner.run();
}},
{"cudnn_conv", [](Op* op) {
    struct ConvOp : Op {
        Var* x, * w, * y;
        int strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups;
        string xformat, wformat, yformat;
        void run_acl() {
            AclOpRunner runner("Conv2D");
            runner.add(x, true, ACL_FORMAT_NCHW);
            runner.add(w, true, ACL_FORMAT_NCHW);
            runner.add(y, false, ACL_FORMAT_NCHW);
            runner.set_attr("strides", vector<int64_t>{1,1,strideh,stridew});
            runner.set_attr("pads", vector<int64_t>{paddingh,paddingh,paddingw,paddingw});
            runner.set_attr("dilations", vector<int64_t>{1,1,dilationh,dilationw});
            runner.set_attr("groups", groups);
            ASSERT(xformat=="abcd" && yformat=="abcd" && wformat=="oihw");
            runner.set_attr("data_format", "NCHW");
            runner.run();
        }
    };
    auto _op = (ConvOp*)op;
    _op->run_acl();
}},
{"cudnn_conv_backward_x", [](Op* op) {
    struct ConvBackwardXOp : Op {
    Var* w, * dy, * dx;
    int xh, xw, strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups;
    string xformat, wformat, yformat;
        void run_acl() {
            AclOpRunner runner("Conv2DBackpropInput");
            runner.add_input_host_nv32(dx->shape); // 10,3,50,50
            // runner.add_input_host_nv32(dy->shape); // 10,3,50,50
            runner.add(w, true, ACL_FORMAT_NCHW); // 4,3,3,3
            aclSetTensorDescName(runner.input_desc.back(), "filter");
            runner.add(dy, true, ACL_FORMAT_NCHW); // 10,4,48,48
            aclSetTensorDescName(runner.input_desc.back(), "out_backprop");
            runner.add(dx, false, ACL_FORMAT_NCHW); // 10,3,50,50
            aclSetTensorDescName(runner.input_desc.back(), "y");
            runner.set_attr("strides", vector<int64_t>{1,1,strideh,stridew});
            runner.set_attr("pads", vector<int64_t>{paddingh,paddingh,paddingw,paddingw});
            runner.set_attr("dilations", vector<int64_t>{1,1,dilationh,dilationw});
            runner.set_attr("groups", groups);
            runner.set_attr("data_format", "NCHW");
            // runner.set_attr("dataFormat", "NCHW");
            // runner.set_attr("data_format", "NCHW");
            ASSERT(xformat=="abcd" && yformat=="abcd" && wformat=="oihw");
            runner.run();
        }
    };
    auto _op = (ConvBackwardXOp*)op;
    _op->run_acl();
}},
{"cudnn_conv_backward_w", [](Op* op) {
    struct ConvBackwardWOp : Op {
    Var* x, * dy, * dw;
    int kh, kw, strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups;
    string xformat, wformat, yformat;
        void run_acl() {
            AclOpRunner runner("Conv2DBackpropFilter");
            runner.add(x, true, ACL_FORMAT_NCHW);
            runner.add_input_host_nv32(dw->shape);
            runner.add(dy, true, ACL_FORMAT_NCHW);
            runner.add(dw, false, ACL_FORMAT_NCHW);
            runner.set_attr("strides", vector<int64_t>{1,1,strideh,stridew});
            runner.set_attr("pads", vector<int64_t>{paddingh,paddingh,paddingw,paddingw});
            runner.set_attr("dilations", vector<int64_t>{1,1,dilationh,dilationw});
            runner.set_attr("groups", groups);
            runner.set_attr("data_format", "NCHW");
            // runner.set_attr("dataFormat", "NCHW");
            // runner.set_attr("data_format", "NCHW");
            // runner.set_attr("data_origin_format", "NCHW");
            ASSERT(xformat=="abcd" && yformat=="abcd" && wformat=="oihw");
            runner.run();
        }
    };
    auto _op = (ConvBackwardWOp*)op;
    _op->run_acl();
}},
// {"cub_arg_reduce", }
};

static void exec_mapped_acl_ops(Op* op) {
    auto iter = acl_ops.find(op->name());
    if (iter != acl_ops.end()) {
        LOGv << "exec acl op " << op->name() << op;
        iter->second(op);
    } else {
        LOGf << "op " << op->name() << " not supported";
    }
}

static jit_op_entry_t acl_do_compile(Op* op) {
    LOGv << "compile" << op;
    OpCompiler oc(op);
    string* src = &oc.src;
    for (auto op_type : op_types)
        op_type->post_pass(&oc);
    string src_after_passes;
    // if is fused op
    if (oc.op) {
        TunerManager tm(&oc);
        src_after_passes = tm.tune();
        src = &src_after_passes;
    }
    op->compile_optimize(*src);
    if (!op->flags.get(NodeFlags::_cuda)) {
        LOGv << "compile cpu";
        return oc.compile(op->get_jit_key(get_jk()), *src);
    }
    if (op->name() == string("fused")) {
        FusedOp* fop = (FusedOp*)op; 
        // if is a relayed op
        if (fop->context->vrm.relay_groups.size()) {
            LOGv << "relay fused op";
            return oc.compile(op->get_jit_key(get_jk()), *src);
        } else {
            return &try_exec_and_fallback_cpu;
        }
    } else 
    if (op->name() == string("code")) {
        CodeOp* cop = (CodeOp*)op;
        if (cop->cuda_src.find("acl") != string::npos) {
            LOGv << "compile acl op";
            return oc.compile(op->get_jit_key(get_jk()), *src);
        } else {
            return &exec_mapped_acl_ops;
        }
    } else
    {
        LOGv << "compile finish" << op;
        return &exec_mapped_acl_ops;
    }
    return do_compile_inner(op);
}

// from op_register.cc
extern unordered_map<string, OpInfo> op_info_map;

void init_acl_ops() {
    do_compile_hook = acl_do_compile;
    vector<string> to_erase;
    for (auto& kv : op_info_map) {
        if (startswith(kv.first, "cu") && acl_ops.count(kv.first) == 0) {
            to_erase.push_back(kv.first);
        }
    }
    for (auto& k : to_erase) {
        LOGv << "op not supported: " << k << ", erase it.";
        op_info_map.erase(k);
    }
}


} // jittor