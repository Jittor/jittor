#include "indexing_codegen.h"
#include "ops/getitem_op.h"
#include "ops/setitem_op.h"
#include "opt/kernel_ir.h"
#include "utils/str_utils.h"

namespace jittor {

void cuda_indexing_optimize(Op* op, NanoVector o_shape, string& src) {
    auto jd = op->get_jit_define();
    map<string,string> jd_map(jd.begin(), jd.end());

    KernelIR main(src);
    auto& func = main.children.back()->children.back();
    // auto& loop = func->children.back();

    func->push_back("void slice_func() {}", &func->before);

    auto& new_func = func->before.back();
    // auto new_func = func->before.back()->move_out();

    new_func->attrs[kir::dtype] = "static __global__ void";
    // LOGir << main.to_string();
    src = main.to_string();
    string arg_call = "";
    const char* tname[] = {"threadIdx.x", "threadIdx.y", "threadIdx.z", "blockIdx.x", "blockIdx.y", "blockIdx.z"};
    const char* tname2[] = {"blockDim.x", "blockDim.y", "blockDim.z", "gridDim.x", "gridDim.y", "gridDim.z"};
    for (auto& ir : func->children) {
        if (ir->type == KernelIRType::define) {
            string& rvalue = ir->require_attr(kir::rvalue);
            string& lvalue = ir->require_attr(kir::lvalue);
            string& dtype = ir->require_attr(kir::dtype);
            if (startswith(rvalue, "input")
                || startswith(rvalue, "output")
                || startswith(rvalue, "vs.")
                || rvalue.back() == ')'
                || rvalue.back() == ']')
            {
                if (dtype == "auto")
                    LOGvvvv << "keep" << rvalue;
                else {
                    LOGvvvv << "args" << rvalue;
                    if (arg_call.size()) arg_call += ", ";
                    arg_call += lvalue;
                    LOGvvvv << dtype+" "+lvalue;
                    new_func->push_back(dtype+" "+lvalue+";", &new_func->inner);
                }
            } else {
                LOGvvvv << "move" <<rvalue;
                new_func->push_back(ir->clone());
            }
        }
    }
    new_func->push_back(func->children.back()->move_out());
    auto& loop = new_func->children.back();
    int no = o_shape.size();
    STACK_ALLOC(KernelIR*, loops, no);
    if (!no) {
        func->push_back("slice_func<<<1,1>>>("+arg_call+");");
    } else {
        bool has_zero = 0;
        loops[0] = loop.get();
        for (int i=1; i<no; i++)
            loops[i] = loops[i-1]->children.back().get();
        for (int i=0; i<no; i++) {
            auto l = loops[i];
            ASSERT(l->inner.size() == 3);
            auto lo = l->find_define("LO"+S(i));
            ASSERT(lo);
            auto loi = std::stoi(lo->require_attr(kir::rvalue));
            if (loi>>7) has_zero = 1;
            string tid = "";
            string tnum = "";
            for (int j=0; j<6; j++) {
                if ((loi>>j)&1) {
                    if (tid.size()) {
                        tid += string("+")+tnum+"*"+tname[j];
                        tnum += string("*")+tname2[j];
                    } else {
                        tid = tname[j];
                        tnum = tname2[j];
                    }
                }
            }
            if (!tid.size()) {
                continue;
            }
            if (loi&(1<<6)) {
                l->inner.at(0)->require_attr(kir::rvalue) = tid;
                l->inner.at(2)->require_attr(kir::code) = "i"+S(i)+"+="+tnum+";";
            } else {
                // no need for
                while (l->inner.size())
                    l->inner.at(0)->erase();
                l->push_front("index_t i"+S(i)+" = "+tid+";");
            }
        }
        if (!has_zero) {
            func->push_back("int no = o_shape.size();");
            func->push_back("STACK_ALLOC(int,masks,no);");
            func->push_back("int tdims[6];");
            func->push_back("cuda_loop_schedule(o_shape, masks, tdims);");
            func->push_back("dim3 grid_dim(tdims[3],tdims[4],tdims[5]);");
            func->push_back("dim3 block_dim(tdims[0],tdims[1],tdims[2]);");
            func->push_back("slice_func<<<grid_dim, block_dim>>>("+arg_call+");");
        }
    }
    src = main.to_string();
}


static void append_cuda_indexing_key(NanoVector shape, JK& key) {
    int count = shape.size();
    STACK_ALLOC(int, masks, count);
    int dimensions[6];
    cuda_loop_schedule(shape, masks, dimensions);
    for (int i = 0; i < count; ++i)
        key << "«LO" << JK::hex1(i) << '=' << JK::hex(masks[i]);
}

void GetitemOp::configure_accelerator_codegen(Codegen& codegen) {
    codegen.fragment = [](Op* op, JK& key) {
        auto* indexing = static_cast<GetitemOp*>(op);
        indexing->GetitemOp::jit_prepare(key);
        append_cuda_indexing_key(indexing->o_shape, key);
    };
    codegen.optimize = [](Op* op, string& source) {
        cuda_indexing_optimize(op, static_cast<GetitemOp*>(op)->o_shape, source);
    };
}

void SetitemOp::configure_accelerator_codegen(Codegen& codegen) {
    codegen.fragment = [](Op* op, JK& key) {
        auto* indexing = static_cast<SetitemOp*>(op);
        indexing->SetitemOp::jit_prepare(key);
        append_cuda_indexing_key(indexing->o_shape, key);
    };
    codegen.optimize = [](Op* op, string& source) {
        cuda_indexing_optimize(op, static_cast<SetitemOp*>(op)->o_shape, source);
    };
}
} // namespace jittor
