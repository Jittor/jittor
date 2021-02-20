// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "var.h"
#include "ops/reindex_op.h"
#include "ops/reindex_reduce_op.h"
#include "ops/reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "opt/tuner/conv_tuner.h"
#include "opt/pass_manager.h"
#include "opt/expr.h"
#include "ops/op_register.h"

#include <algorithm>
#include <cstring>

namespace jittor {

using namespace expr;

struct OpInspector {
    // binary mask for
    // m1: exact dimension map
    // m2: no relation
    // m3: other
    uint64 m1=0, m2=0, m3=0;
    // which dimension map
    vector<int> mm;
    Op* op;
    bool failed=0;

    void init(ReindexOp* op) {
        unordered_map<string, int> p;
        mm.resize(op->y->shape.size(), -1);
        for (uint i=0; i<op->y->shape.size(); i++)
            p["i"+S(i)] = i;
        for (uint i=0; i<op->x->shape.size(); i++) {
            if (p.count(op->indexes[i])) {
                int j = p.at(op->indexes[i]);
                if (mm[j]!=-1) failed=1;
                mm[j] = i;
                m1 |= 1ll<<j;
            } else {
                auto e = expr::make(op->indexes[i]);
                expr::dfs(e.get(), [&](expr::Expr* e) {
                    if (e->is_sym() && p.count(e->str)) {
                        int j = p.at(e->str);
                        if (mm[j]!=-1) failed=1;
                        m3 |= 1ll << j;
                        mm[j] = i;
                    }
                });
            }
        }
        m2 = ((1ll<<mm.size())-1) ^ (m1|m3);
    }

    OpInspector(ReindexOp* op) : op(op) { init(op); }

    void init(ReindexReduceOp* op) {
        unordered_map<string, int> p;
        mm.resize(op->y->shape.size(), -1);
        for (uint i=0; i<op->y->shape.size(); i++)
            p["i"+S(i)] = i;
        for (uint i=0; i<op->x->shape.size(); i++) {
            if (p.count(op->indexes[i])) {
                int j = p.at(op->indexes[i]);
                if (mm[j]!=-1) failed=1;
                mm[j] = i;
                m1 |= 1ll<<j;
            } else {
                auto e = expr::make(op->indexes[i]);
                expr::dfs(e.get(), [&](expr::Expr* e) {
                    if (e->is_sym() && p.count(e->str)) {
                        int j = p.at(e->str);
                        if (mm[j]!=-1) failed=1;
                        m3 |= 1ll << j;
                        mm[j] = i;
                    }
                });
            }
        }
        m2 = ((1ll<<mm.size())-1) ^ (m1|m3);
    }

    OpInspector(ReindexReduceOp* op) : op(op) { init(op); }

    void init(BroadcastToOp* op) {
        mm.resize(op->z->shape.size(), 0);
        m2 = op->bcast_mask;
        m1 =  ((1ll<<mm.size())-1) ^ (m2);
        for (uint i=0,j=0; i<op->z->shape.size(); i++)
            if ((m1>>i)&1) mm[i] = j++;
    }

    OpInspector(BroadcastToOp* op) : op(op) { init(op); }

    void init(ReduceOp* op) {
        mm.resize(op->x->shape.size(), 0);
        m2 = op->reduce_mask;
        m1 =  ((1ll<<mm.size())-1) ^ (m2);
        for (uint i=0,j=0; i<op->x->shape.size(); i++)
            if ((m1>>i)&1) mm[i] = j++;
    }

    OpInspector(ReduceOp* op) : op(op) { init(op); }

    OpInspector(Op* op) : op(op) {
        if (strcmp(op->name(), "reduce") == 0)
            init((ReduceOp*)op);
        else if (strcmp(op->name(), "broadcast_to") == 0)
            init((BroadcastToOp*)op);
        else if (strcmp(op->name(), "reindex") == 0)
            init((ReindexOp*)op);
        else if (strcmp(op->name(), "reindex_reduce") == 0)
            init((ReindexReduceOp*)op);
        else
            failed = 1;
    }
    
    // get last one index of binary mask
    void get_id(uint64 m, int& i) {
        if (m==0) failed=1;
        else {
            i=0;
            while (!(m&1)) i++,m>>=1;
            if (m!=1) failed=1;
        }
    }
    // get last two index of binary mask
    void get_id(uint64 m, int& i, int& j) {
        if (m==0) failed=1;
        else {
            i=j=0;
            while (!(m&1)) i++,m>>=1;
            if (m<=1) {
                failed=1;
                return;
            }
            j=i+1,m>>=1;
            while (!(m&1)) j++,m>>=1;
            if (m!=1) failed=1;
        }
    }

    // get last three index of binary mask
    void get_id(uint64 m, int& i, int& j, int& k) {
        if (m==0) failed=1;
        else {
            i=j=0;
            while (!(m&1)) i++,m>>=1;
            if (m<=1) {
                failed=1;
                return;
            }
            j=i+1,m>>=1;
            while (!(m&1)) j++,m>>=1;
            if (m<=1) {
                failed=1;
                return;
            }
            k=j+1, m>>=1;
            while (!(m&1)) k++,m>>=1;
            if (m!=1) {
                failed=1;
                return;
            }
        }
    }    

    bool check_overlap(const vector<int>& v) {
        uint64 sum=0;
        for (auto a : v) {
            if (sum & (1ll<<a)) return failed=1;
            sum |= 1ll<<a;
        }
        return 0;
    }

    string format(const string& fmt, const vector<int>& order) {
        string new_fmt = fmt;
        if (order.size() != fmt.size()) {
            failed = 1;
            return "";
        }
        if (check_overlap(order))
            return "";
        for (uint i=0; i<order.size(); i++) {
            if (order[i]>=(int)new_fmt.size()) {
                failed = 1;
                return "";
            }
            new_fmt[order[i]] = fmt[i];
        }
        return new_fmt;
    }
};

std::ostream& operator<<(std::ostream& os, const OpInspector& oi) {
    if (oi.failed) return os << "inspect failed";
    for (uint i=0; i<oi.mm.size(); i++) os << ((oi.m1>>i)&1);
    os << ',';
    for (uint i=0; i<oi.mm.size(); i++) os << ((oi.m2>>i)&1);
    os << ',';
    for (uint i=0; i<oi.mm.size(); i++) os << ((oi.m3>>i)&1);
    return os << ',' << oi.mm;
}

void ConvTuner::forwardTune(FusedOp* fop) {
    for (Op* op : fop->ops)
    if (op->name_ex()=="reduce.add" || op->name_ex()=="reindex_reduce.add") {
        // reduce op and reindex reduce op have the same memory layout
        // it is ok to force cast.
        auto op_iop = op->input(0)->input();
        if (!(op_iop
            && op_iop->name_ex()=="binary.multiply"
            && fop->has(op_iop)))
            continue;
        auto bop = (BinaryOp*)op_iop;

        if (!(bop->y->input() && bop->x->input() && fop->has(bop->x->input()) && fop->has(bop->y->input()))) continue;
        if (!(bop->x->input()->type()==OpType::broadcast && bop->y->input()->type()==OpType::broadcast)) return;

        // only support float32 currently
        if (bop->z->dtype() != ns_float32)
            continue;
        Op* ops[3] = {op, bop->x->input(), bop->y->input()};
        int ok = 0;
        LOGvvvv << "conv like op" << fop << fop->get_jit_key(jk);
        for (int y_id=0; y_id<3; y_id++)
        for (int x_id=0; x_id<3; x_id++)
        for (int w_id=0; w_id<3; w_id++) {
            if (ok) break;
            if (x_id == y_id || x_id == w_id || y_id == w_id) continue;
            LOGvvvv << "try" << x_id << y_id << w_id;
            OpInspector xoi(ops[x_id]);
            OpInspector yoi(ops[y_id]);
            OpInspector woi(ops[w_id]);
            vector<string>* xop_indexes;
            if (strcmp(xoi.op->name(), "reindex") == 0) {
                xop_indexes = &((ReindexOp*)xoi.op)->indexes;
            } else
            if (strcmp(xoi.op->name(), "reindex_reduce") == 0) {
                xop_indexes = &((ReindexReduceOp*)xoi.op)->indexes;
            } else
                continue;
            if (xoi.failed || yoi.failed || woi.failed) continue;
            int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
            int zn, zg, zci, zco, zh, zw, zwh, zww;
            zn = zci = zco = zh = zw = zwh = zww = 0;
            if (bop->x->shape.size() == 7) {
                xoi.get_id(xoi.m1 & woi.m2 & yoi.m1, zn);
                xoi.get_id(xoi.m1 & woi.m1 & yoi.m2, zci);
                xoi.get_id(xoi.m3 & woi.m2 & yoi.m1, zh, zw);
                xoi.get_id(xoi.m2 & woi.m1 & yoi.m1, zco);
                xoi.get_id(xoi.m3 & woi.m1 & yoi.m2, zwh, zww);
                LOGvvvv << "zn,zci,zco,zh,zw,zwh,zww =" << vector<int>{zn,zci,zco,zh,zw,zwh,zww};
                xoi.check_overlap({zn,zci,zco,zh,zw,zwh,zww});
                zg = -1;
            } else {
                if (bop->x->shape.size() != 8)
                    continue;
                // group conv
                xoi.get_id(xoi.m1 & woi.m2 & yoi.m1, zn);
                xoi.get_id(xoi.m3 & woi.m3 & yoi.m3, zg);
                xoi.get_id(xoi.m3 & woi.m2 & yoi.m1, zh, zw);
                xoi.get_id(xoi.m2 & woi.m3 & yoi.m3, zco);
                xoi.get_id(xoi.m3 & woi.m1 & yoi.m2, zci, zwh, zww);
                LOGvvvv << "zn,zg,zci,zco,zh,zw,zwh,zww =" << vector<int>{zn,zg,zci,zco,zh,zw,zwh,zww};
                xoi.check_overlap({zn,zg,zci,zco,zh,zw,zwh,zww});
            }
            if (xoi.failed) continue;
            xn = xoi.mm[zn];
            xc = xoi.mm[zci];
            xh = xoi.mm[zh];
            xw = xoi.mm[zw];
            LOGvvvv << "xnchw =" << vector<int>{xn,xc,xh,xw};
            auto xformat = xoi.format("abcd", {xn, xc, xh, xw});
            LOGvvvv << "xformat =" << xformat;
            wci = woi.mm[zci];
            wco = woi.mm[zco];
            wh = woi.mm[zwh];
            ww = woi.mm[zww];
            auto wformat = xoi.format("iohw", {wci, wco, wh, ww});
            LOGvvvv << "wformat =" << wformat;
            yn = yoi.mm[zn];
            yc = yoi.mm[zco];
            yh = yoi.mm[zh];
            yw = yoi.mm[zw];
            auto yformat = xoi.format("abcd", {yn, yc, yh, yw});
            LOGvvvv << "yformat =" << yformat;

            // mkl doesn't support "cdab" format
            if (yformat == "cdab") continue;
            // cuda doesn't support "iohw" format
            if (fop->flags.get(NodeFlags::_cuda) && wformat == "iohw") continue;
            if (xoi.failed) continue;
            std::stringstream ss;
            // i@zh*stride+i@zwh+padding
            ss << "i" << zh << "*stride+i" << zwh << "*dilation+padding";
            auto expr_h = expr::make(ss.str());
            ss.str("");
            ss << "i" << zw << "*stride+i" << zww << "*dilation+padding";
            auto expr_w = expr::make(ss.str());

            vector<unique_ptr<Expr>> rh, rw;
            auto src_h = expr::make(xop_indexes->at(xh));
            if (!expr::match(src_h.get(), expr_h.get(), {"stride", "padding", "dilation"}, {"i"+S(zh), "i"+S(zwh)}, rh)) {
                LOGvvvv << "Expr not match" << src_h << expr_h;
                continue;
            }
            LOGvvvv << "H Expr matched" << src_h << expr_h;
            if (!rh[0]->is(expr::_number) || !rh[1]->is(expr::_number) || !rh[2]->is(expr::_number)) return;
            auto src_w = expr::make(xop_indexes->at(xw));
            if (!expr::match(src_w.get(), expr_w.get(), {"stride", "padding", "dilation"}, {"i"+S(zw), "i"+S(zww)}, rw))
                continue;
            LOGvvvv << "W Expr matched" << src_w << expr_w;
            if (!rw[0]->is(expr::_number) || !rw[1]->is(expr::_number) || !rw[2]->is(expr::_number)) return;
            int stride_h = rh[0]->as_int();
            int padding_h = -rh[1]->as_int();
            int dilation_h = rh[2]->as_int();
            int stride_w = rw[0]->as_int();
            int padding_w = -rw[1]->as_int();
            int dilation_w = rw[2]->as_int();
            if (dilation_h < 1 || dilation_w < 1) continue;
            LOGvvvv <<  "get stride padding and dilation" << stride_h << padding_h << dilation_h;
            if (xformat == "bacd") {
                LOGvvvv << "mkl not support bacd, continue";
                continue;
            }
            Var* x = x_id == 0 ? xoi.op->output(0) : xoi.op->input(0);
            Var* w = w_id == 0 ? woi.op->output(0) : woi.op->input(0);
            Var* y = y_id == 0 ? yoi.op->output(0) : yoi.op->input(0);

            int oh = (x->shape[xh]-w->shape[wh]*dilation_h+dilation_h-1+padding_h*2)/stride_h+1;
            int ow = (x->shape[xw]-w->shape[ww]*dilation_w+dilation_w-1+padding_w*2)/stride_w+1;
            if (oh != y->shape[yh] || ow != y->shape[yw]) {
                LOGvvvv << "shape not match" << "(" >> oh >> "," >> ow >> ") !="
                    << "(" >> y->shape[yh] >> "," >> y->shape[yw] >> ")";
                continue;
            }
            int groups = zg==-1 ? 1 : x->shape[xc] / w->shape[wci];
            LOGvvvv << "groups: " << groups;
            if (groups>1 && wformat != "oihw")
                continue;

            VarPtr rvar;
            int rid;
            string relay_conv_name;

            if (y_id == 0) {
                relay_conv_name = fop->flags.get(NodeFlags::_cpu) ?
                    "mkl_conv" : "cudnn_conv";
                if (!has_op(relay_conv_name))
                    continue;
                auto make_conv = get_op_info(relay_conv_name)
                        .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, string, string, string>();
                LOGvvvv << x << w << stride_h << stride_w << padding_h << padding_w << dilation_h << dilation_w << groups << xformat << wformat << yformat;
                rvar = make_conv(x, w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, xformat, wformat, yformat);
            } else
            if (x_id == 0) {
                relay_conv_name = fop->flags.get(NodeFlags::_cpu) ?
                        "mkl_conv_backward_x" : "cudnn_conv_backward_x";
                if (!has_op(relay_conv_name))
                    continue;
                auto height = x->shape[xformat.find("c")];
                auto width = x->shape[xformat.find("d")];
                auto make_conv_x = get_op_info(relay_conv_name)
                        .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, string, string, string>();
                LOGvvvv << w << y << height << width << stride_h << stride_w << padding_h << padding_w << dilation_h << dilation_w << groups << xformat << wformat << yformat;
                rvar = make_conv_x(w, y, height, width, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, xformat, wformat, yformat);
            } else {
                relay_conv_name = fop->flags.get(NodeFlags::_cpu) ?
                        "mkl_conv_backward_w" : "cudnn_conv_backward_w";
                if (!has_op(relay_conv_name))
                    continue;
                auto kh = w->shape[wformat.find("h")];
                auto kw = w->shape[wformat.find("w")];
                LOGvvvv << x << y << kh << stride_h << stride_w << padding_h << padding_w << dilation_h << dilation_w << groups << xformat << wformat << yformat;
                auto make_conv_w = get_op_info(relay_conv_name)
                        .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, string, string, string>();
                rvar = make_conv_w(x, y, kh, kw, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, xformat, wformat, yformat);
            }

            LOGvvvv << relay_conv_name << "output:" << rvar;
            rid = fop->context->vrm.add_relay_group({{rvar, op->output(0)}});
            if (rid>=0) {
                auto srid = "relay"+S(rid);
                add_candidate(srid, 1);
                add_candidate(srid, 0);
                confidence = 20;
                ok = 1;
                LOGvvvv << "ok" << x_id << y_id << w_id;
            }
        }
        }
}

void ConvTuner::run(PassManager* pm, TunerManager* tm) {
    FusedOp* fop=tm->oc->op;

    forwardTune(fop);
}

}
