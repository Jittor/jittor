// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
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
    bool failed=0;
    OpInspector(ReindexOp* op) {
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

    OpInspector(ReindexReduceOp* op) {
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

    OpInspector(BroadcastToOp* op) {
        mm.resize(op->z->shape.size(), 0);
        m2 = op->bcast_mask;
        m1 =  ((1ll<<mm.size())-1) ^ (m2);
        for (uint i=0,j=0; i<op->z->shape.size(); i++)
            if ((m1>>i)&1) mm[i] = j++;
    }

    OpInspector(ReduceOp* op) {
        mm.resize(op->x->shape.size(), 0);
        m2 = op->reduce_mask;
        m1 =  ((1ll<<mm.size())-1) ^ (m2);
        for (uint i=0,j=0; i<op->x->shape.size(); i++)
            if ((m1>>i)&1) mm[i] = j++;
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
        vector<pair<int, int>> order_;
        for (uint i = 0; i < order.size(); i++) {
            order_.push_back(pair<int, int>(order[i], i));
        }
        sort(order_.begin(), order_.end());
        for (uint i=0; i<order_.size(); i++) {
            if (order_[i].second>=(int)new_fmt.size()) {
                failed = 1;
                return "";
            }
            new_fmt[order_[i].second] = fmt[i];
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
    if (op->name_ex()=="reduce.add") {
        auto rop = (ReduceOp*)op;
        if (!(rop->x->input() && rop->x->input()->name_ex()=="binary.multiply" && rop->x->input()->tflag==op->tflag))
            continue;
        auto bop = (BinaryOp*)(rop->x->input());

        if (!(bop->y->input() && bop->x->input() && bop->x->input()->tflag==op->tflag && bop->y->input()->tflag==op->tflag)) continue;
        if (!((bop->x->input()->name_ex()=="reindex" && bop->y->input()->name_ex()=="broadcast_to") || 
        (bop->y->input()->name_ex()=="reindex" && bop->x->input()->name_ex()=="broadcast_to"))) return;
        // riop1 reindex -> xx
        // riop2 broadcast -> ww
        auto riop1 = bop->x->input()->name_ex()=="reindex" ?
            (ReindexOp*)(bop->x->input()) : (ReindexOp*)(bop->y->input());
        auto riop2 = bop->x->input()->name_ex()=="reindex" ?
            (BroadcastToOp*)(bop->y->input()) : (BroadcastToOp*)(bop->x->input());
        LOGvvvv << "conv like op" << fop << fop->get_jit_key();
        OpInspector xoi(riop1);
        LOGvvvv << "inspect x:" << xoi << riop1->indexes;
        OpInspector woi(riop2);
        LOGvvvv << "inspect w:" << woi;
        OpInspector yoi(rop);
        LOGvvvv << "inspect y:" << yoi;
        int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
        int zn, zci, zco, zh, zw, zwh, zww;
        zn = zci = zco = zh = zw = zwh = zww = 0;
        xoi.get_id(xoi.m1 & woi.m2 & yoi.m1, zn);
        xoi.get_id(xoi.m1 & woi.m1 & yoi.m2, zci);
        xoi.get_id(xoi.m3 & woi.m2 & yoi.m1, zh, zw);
        xoi.get_id(xoi.m2 & woi.m1 & yoi.m1, zco);
        xoi.get_id(xoi.m3 & woi.m1 & yoi.m2, zwh, zww);
        LOGvvvv << "zn,zci,zco,zh,zw,zwh,zww =" << vector<int>{zn,zci,zco,zh,zw,zwh,zww};
        xoi.check_overlap({zn,zci,zco,zh,zw,zwh,zww});
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
        auto src_h = expr::make(riop1->indexes[xh]);
        if (!expr::match(src_h.get(), expr_h.get(), {"stride", "padding", "dilation"}, {"i"+S(zh), "i"+S(zwh)}, rh)) {
            LOGvvvv << "Expr not match" << src_h << expr_h;
            continue;
        }
        LOGvvvv << "H Expr matched" << src_h << expr_h;
        if (!rh[0]->is(expr::_number) || !rh[1]->is(expr::_number) || !rh[2]->is(expr::_number)) return;
        auto src_w = expr::make(riop1->indexes[xw]);
        if (!expr::match(src_w.get(), expr_w.get(), {"stride", "padding", "dilation"}, {"i"+S(zw), "i"+S(zww)}, rw))
            return;
        LOGvvvv << "W Expr matched" << src_w << expr_w;
        if (!rw[0]->is(expr::_number) || !rw[1]->is(expr::_number) || !rw[2]->is(expr::_number)) return;
        int stride_h = rh[0]->as_int();
        int padding_h = -rh[1]->as_int();
        int dilation_h = rh[2]->as_int();
        int stride_w = rw[0]->as_int();
        int padding_w = -rw[1]->as_int();
        int dilation_w = rw[2]->as_int();
        if (dilation_h < 1 || dilation_w < 1) continue;
        if (stride_h!=stride_w || padding_h!=padding_w || dilation_h!=dilation_w) {
            LOGvvvv << "cannot relay different stride and padding between h and w"
                << stride_h << padding_h << dilation_h << stride_w << padding_w << dilation_w;
            continue;
        }
        LOGvvvv <<  "get stride padding and dilation" << stride_h << padding_h << dilation_h;
        if (xformat == "bacd" && dilation_h != 1) {
            LOGvvvv << "mkl not support bacd dilation, continue";
            continue;
        }
        int stride = stride_h;
        int padding = padding_h;
        int dilation = dilation_h;
        Var* x = riop1->x;
        Var* w = riop2->x;

        int oh = (x->shape[xh]-w->shape[wh]*dilation_h+dilation_h-1+padding_h*2)/stride_h+1;
        int ow = (x->shape[xw]-w->shape[ww]*dilation_w+dilation_w-1+padding_w*2)/stride_w+1;
        if (oh != rop->y->shape[yh] || ow != rop->y->shape[yw]) continue;

        string relay_conv_name = fop->flags.get(NodeFlags::_cpu) ?
            "mkl_conv" : "cudnn_conv";
        if (!has_op(relay_conv_name))
            continue;
        auto make_conv = get_op_info(relay_conv_name)
                .get_constructor<VarPtr, Var*, Var*, int, int, int, int, string, string, string>();
            auto rvar = make_conv(x, w, stride, padding, dilation, 1, xformat, wformat, yformat);
        auto rid = fop->context->vrm.add_relay_group({{rvar, rop->y}});
        if (rid>=0) {
            auto srid = "relay"+S(rid);
            add_candidate(srid, 1);
            add_candidate(srid, 0);
            confidence = 20;
        }
    }
}

void ConvTuner::backwardTune(FusedOp* fop) {
    for (Op* op : fop->ops) {
    int bo=0;
    Var *x=NULL, *y=NULL, *w=NULL;
    Var *dw=NULL, *dx=NULL;
    int height=0,width=0,kernel_size=0,stride=0, padding=0, dilation=1;
    string xformat, yformat, wformat;
    if (op->name_ex() == "reduce.add") {
        auto rop = (ReduceOp*)op;
        if (!(rop->x->input() && rop->x->input()->name_ex()=="binary.multiply" && rop->x->input()->tflag==op->tflag))
            continue;
        auto bop = (BinaryOp*)(rop->x->input());

        if (!(bop->y->input() && bop->x->input() && bop->x->input()->tflag==op->tflag && bop->y->input()->tflag==op->tflag)) continue;
        if (!((bop->x->input()->name_ex()=="reindex" && bop->y->input()->name_ex()=="broadcast_to") || 
        (bop->y->input()->name_ex()=="reindex" && bop->x->input()->name_ex()=="broadcast_to"))) continue;
        auto riop1 = bop->x->input()->name_ex()=="reindex" ? (ReindexOp*)(bop->x->input()) : (ReindexOp*)(bop->y->input());
        auto riop2 = bop->x->input()->name_ex()=="reindex" ? (BroadcastToOp*)(bop->y->input()) : (BroadcastToOp*)(bop->x->input());
        
        OpInspector xoi(riop1);
        LOGvvvv << "inspect x:" << xoi << riop1->indexes;
        OpInspector yoi(riop2);
        LOGvvvv << "inspect y:" << yoi;
        OpInspector woi(rop);
        LOGvvvv << "inspect w:" << woi;
        int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
        int zn, zci, zco, zh, zw, zwh, zww;
        zn = zci = zco = zh = zw = zwh = zww = 0;
        xoi.get_id(xoi.m1 & woi.m2 & yoi.m1, zn);
        xoi.get_id(xoi.m1 & woi.m1 & yoi.m2, zci);
        xoi.get_id(xoi.m3 & woi.m2 & yoi.m1, zh, zw);
        xoi.get_id(xoi.m2 & woi.m1 & yoi.m1, zco);
        xoi.get_id(xoi.m3 & woi.m1 & yoi.m2, zwh, zww);
        LOGvvvv << "zn,zci,zco,zh,zw,zwh,zww =" << vector<int>{zn,zci,zco,zh,zw,zwh,zww};
        xoi.check_overlap({zn,zci,zco,zh,zw,zwh,zww});
        if (xoi.failed) continue;
        xn = xoi.mm[zn];
        xc = xoi.mm[zci];
        xh = xoi.mm[zh];
        xw = xoi.mm[zw];
        LOGvvvv << "xnchw =" << vector<int>{xn,xc,xh,xw};
        xformat = xoi.format("abcd", {xn, xc, xh, xw});
        LOGvvvv << "xformat =" << xformat;
        wci = woi.mm[zci];
        wco = woi.mm[zco];
        wh = woi.mm[zwh];
        ww = woi.mm[zww];
        wformat = xoi.format("iohw", {wci, wco, wh, ww});
        LOGvvvv << "wformat =" << wformat;
        yn = yoi.mm[zn];
        yc = yoi.mm[zco];
        yh = yoi.mm[zh];
        yw = yoi.mm[zw];
        yformat = xoi.format("abcd", {yn, yc, yh, yw});
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

        vector<unique_ptr<Expr>> rh;
        auto src_h = expr::make(riop1->indexes[xh]);
        if (!expr::match(src_h.get(), expr_h.get(), {"stride", "padding", "dilation"}, {"i"+S(zh), "i"+S(zwh)}, rh)) {
            LOGvvvv << "Expr not match" << src_h << expr_h;
            continue;
        }
        if (!rh[0]->is(expr::_number) || !rh[1]->is(expr::_number)) continue;

        dw = rop->y;
        stride = rh[0]->as_int();
        padding = -rh[1]->as_int();
        dilation = rh[2]->as_int();
        kernel_size = dw->shape[wformat.find("h")];
        x = riop1->x;
        y = riop2->x;
        bo++;
        LOGvvvv <<  "backward_w get stride padding and dilation" << stride << padding << dilation;
    } else if (op->name_ex() == "reindex_reduce.add") {
        auto rop = (ReindexReduceOp*)op;
        if (!(rop->y->input() && rop->y->input()->name_ex()=="binary.multiply" && rop->x->input()->tflag==op->tflag))
            continue;
        auto bop = (BinaryOp*)(rop->y->input());
        
        if (!(bop->y->input() && bop->x->input() && bop->x->input()->tflag==op->tflag && bop->y->input()->tflag==op->tflag)) continue;
        if (!((bop->x->input()->name_ex()=="broadcast_to" && bop->y->input()->name_ex()=="broadcast_to"))) return;
        auto riop1 = (BroadcastToOp*)(bop->x->input());
        auto riop2 = (BroadcastToOp*)(bop->y->input());
        
        OpInspector woi(riop1);
        LOGvvvv << "inspect w:" << woi;
        OpInspector yoi(riop2);
        LOGvvvv << "inspect y:" << yoi;
        OpInspector xoi(rop);
        LOGvvvv << "inspect x:" << xoi;
        int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
        int zn, zci, zco, zh, zw, zwh, zww;
        zn = zci = zco = zh = zw = zwh = zww = 0;
        xoi.get_id(xoi.m1 & woi.m2 & yoi.m1, zn);
        xoi.get_id(xoi.m1 & woi.m1 & yoi.m2, zci);
        xoi.get_id(xoi.m3 & woi.m2 & yoi.m1, zh, zw);
        xoi.get_id(xoi.m2 & woi.m1 & yoi.m1, zco);
        xoi.get_id(xoi.m3 & woi.m1 & yoi.m2, zwh, zww);
        LOGvvvv << "zn,zci,zco,zh,zw,zwh,zww =" << vector<int>{zn,zci,zco,zh,zw,zwh,zww};
        xoi.check_overlap({zn,zci,zco,zh,zw,zwh,zww});
        if (xoi.failed) continue;
        xn = xoi.mm[zn];
        xc = xoi.mm[zci];
        xh = xoi.mm[zh];
        xw = xoi.mm[zw];
        LOGvvvv << "xnchw =" << vector<int>{xn,xc,xh,xw};
        xformat = xoi.format("abcd", {xn, xc, xh, xw});
        LOGvvvv << "xformat =" << xformat;
        wci = woi.mm[zci];
        wco = woi.mm[zco];
        wh = woi.mm[zwh];
        ww = woi.mm[zww];
        wformat = xoi.format("iohw", {wci, wco, wh, ww});
        LOGvvvv << "wformat =" << wformat;
        yn = yoi.mm[zn];
        yc = yoi.mm[zco];
        yh = yoi.mm[zh];
        yw = yoi.mm[zw];
        yformat = xoi.format("abcd", {yn, yc, yh, yw});
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

        vector<unique_ptr<Expr>> rh;
        auto src_h = expr::make(rop->indexes[xh]);
        if (!expr::match(src_h.get(), expr_h.get(), {"stride", "padding", "dilation"}, {"i"+S(zh), "i"+S(zwh)}, rh)) {
            LOGvvvv << "Expr not match" << src_h << expr_h;
            continue;
        }
        if (!rh[0]->is(expr::_number) || !rh[1]->is(expr::_number)) continue;
        
        dx = rop->x;
        stride = rh[0]->as_int();
        padding = -rh[1]->as_int();
        dilation = rh[2]->as_int();
        height = dx->shape[xformat.find("c")];
        width = dx->shape[xformat.find("d")];
        w = riop1->x;
        y = riop2->x;
        bo+=2;
        LOGvvvv <<  "backward_x get stride padding and dilation" << stride << padding << dilation;
    }

    // TODO: CUDA only support nchw(abcd)
    if (fop->flags.get(NodeFlags::_cuda) && (xformat != "abcd" || yformat != "abcd"))
        continue;

    if (bo&1) {
        auto make_conv_w = get_op_info(
            fop->flags.get(NodeFlags::_cpu) ?
                "mkl_conv_backward_w" : "cudnn_conv_backward_w"
            ).get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, string, string, string>();
        auto rvar_w = make_conv_w(x, y, kernel_size, stride, padding, dilation, 1, xformat, wformat, yformat);
        auto rid = fop->context->vrm.add_relay_group({{rvar_w, dw}});
        if (rid>=0) {
            auto srid = "relay"+S(rid);
            add_candidate(srid, 1);
            add_candidate(srid, 0);
            confidence = 20;
        }
    }
    if (bo&2) {
        auto make_conv_x = get_op_info(
            fop->flags.get(NodeFlags::_cpu) ?
                "mkl_conv_backward_x" : "cudnn_conv_backward_x"
            ).get_constructor<VarPtr, Var*, Var*, int , int, int, int, int, int, string, string, string>();
        auto rvar_x = make_conv_x(w, y, height, width, stride, padding, dilation, 1, xformat, wformat, yformat);
        auto rid = fop->context->vrm.add_relay_group({{rvar_x, dx}});
        if (rid>=0) {
            auto srid = "relay"+S(rid);
            add_candidate(srid, 1);
            add_candidate(srid, 0);
            confidence = 20;
        }
    }
    }
}

void ConvTuner::run(PassManager* pm, TunerManager* tm) {
    FusedOp* fop=tm->oc->op;

    forwardTune(fop);
    backwardTune(fop);
}

void GroupConvTuner::forwardTune(FusedOp* fop) {
    LOGvvvv << "tune group conv";
    for (Op* op : fop->ops) {
        if (op->name_ex()=="reindex_reduce.add") {
            auto rop = (ReindexReduceOp*)op;
            if (!(rop->y->input() && rop->y->input()->name_ex()=="binary.multiply" && rop->y->input()->tflag==op->tflag))
                continue;
            auto bop = (BinaryOp*)(rop->y->input());

            if (!(bop->y->input() && bop->x->input() && bop->x->input()->tflag==op->tflag && bop->y->input()->tflag==op->tflag)) continue;
            if (!(bop->x->input()->name_ex()=="reindex" && bop->y->input()->name_ex()=="reindex")) return;
            auto riop1 = (ReindexOp*)(bop->x->input());
            auto riop2 = (ReindexOp*)(bop->y->input());
            LOGvvvv << "conv like op" << fop << fop->get_jit_key();
            OpInspector xoi(riop1);
            OpInspector woi(riop2);
            // determine which is which (since both are ReindexOp)
            if (xoi.mm[0] == -1 && woi.mm[0] == 0) {
                std::swap(xoi, woi);
            }
            OpInspector yoi(rop);
            int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
            int zn, zg, zci, zco, zh, zw, zwh, zww;
            zn = zg = zci = zco = zh = zw = zwh = zww = 0;
            xoi.get_id(xoi.m1 & woi.m2 & yoi.m1, zn);
            xoi.get_id(xoi.m3 & woi.m3 & yoi.m3, zg);
            xoi.get_id(xoi.m3 & woi.m2 & yoi.m1, zh, zw);
            xoi.get_id(xoi.m2 & woi.m3 & yoi.m3, zco);
            xoi.get_id(xoi.m3 & woi.m1 & yoi.m2, zci, zwh, zww);
            LOGvvvv << "zn,zg,zci,zco,zh,zw,zwh,zww =" << vector<int>{zn,zg,zci,zco,zh,zw,zwh,zww};
            xoi.check_overlap({zn,zg,zci,zco,zh,zw,zwh,zww});
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
            auto src_h = expr::make(riop1->indexes[xh]);
            if (!expr::match(src_h.get(), expr_h.get(), {"stride", "padding", "dilation"}, {"i"+S(zh), "i"+S(zwh)}, rh)) {
                LOGvvvv << "Expr not match" << src_h << expr_h;
                continue;
            }
            if (!rh[0]->is(expr::_number) || !rh[1]->is(expr::_number) || !rh[2]->is(expr::_number)) return;
            auto src_w = expr::make(riop1->indexes[xw]);
            if (!expr::match(src_w.get(), expr_w.get(), {"stride", "padding", "dilation"}, {"i"+S(zw), "i"+S(zww)}, rw))
                return;
            if (!rw[0]->is(expr::_number) || !rw[1]->is(expr::_number) || !rw[2]->is(expr::_number)) return;
            int stride_h = rh[0]->as_int();
            int padding_h = -rh[1]->as_int();
            int dilation_h = rh[2]->as_int();
            int stride_w = rw[0]->as_int();
            int padding_w = -rw[1]->as_int();
            int dilation_w = rw[2]->as_int();
            if (dilation_h < 1 || dilation_w < 1) continue;
            if (stride_h!=stride_w || padding_h!=padding_w || dilation_h!=dilation_w) {
                LOGvvvv << "cannot relay different stride and padding between h and w"
                    << stride_h << padding_h << dilation_h << stride_w << padding_w << dilation_w;
                continue;
            }
            LOGvvvv <<  "get stride padding and dilation" << stride_h << padding_h << dilation_h;

            int stride = stride_h;
            int padding = padding_h;
            int dilation = dilation_h;
            Var* x = riop1->x;
            Var* w = riop2->x;

            int oh = (x->shape[xh]-w->shape[wh]*dilation_h+dilation_h-1+padding_h*2)/stride_h+1;
            int ow = (x->shape[xw]-w->shape[ww]*dilation_w+dilation_w-1+padding_w*2)/stride_w+1;
            if (oh != rop->x->shape[yh] || ow != rop->x->shape[yw]) continue;

            int groups = x->shape[xc] / w->shape[wci];
            LOGvvvv << "groups: " << groups;
            if (fop->flags.get(NodeFlags::_cpu) && groups > 1) {
                LOGi << "group conv does not support mkl";
                continue;
            }

            string relay_conv_name = "cudnn_conv";
            if (!has_op(relay_conv_name))
                continue;
            auto make_conv = get_op_info(relay_conv_name)
                .get_constructor<VarPtr, Var*, Var*, int, int, int, int, string, string, string>();
            auto rvar = make_conv(x, w, stride, padding, dilation, groups, xformat, wformat, yformat);
            auto rid = fop->context->vrm.add_relay_group({{rvar, rop->x}});
            if (rid>=0) {
                auto srid = "relay"+S(rid);
                add_candidate(srid, 1);
                add_candidate(srid, 0);
                confidence = 20;
            }
        }
    }
}

void GroupConvTuner::backwardTune(FusedOp* fop) {
    for (Op* op : fop->ops) {
        int bo=0;
        Var *x=NULL, *y=NULL, *w=NULL;
        Var *dw=NULL, *dx=NULL;
        int height=0,width=0,kernel_size=0,stride=0, padding=0, dilation=1, groups=1;
        string xformat, yformat, wformat;
        if (op->name_ex() == "reindex_reduce.add") {
            auto rop = (ReindexReduceOp*)op;
            if (!(rop->y->input() && rop->y->input()->name_ex()=="binary.multiply" && rop->y->input()->tflag==op->tflag))
                continue;
            auto bop = (BinaryOp*)(rop->y->input());
            if (!(bop->y->input() && bop->x->input() && bop->x->input()->tflag==op->tflag && bop->y->input()->tflag==op->tflag)) continue;
            if (!(bop->x->input()->name_ex()=="reindex" && bop->y->input()->name_ex()=="reindex")) return;
            auto riop1 = (ReindexOp*)(bop->x->input());
            auto riop2 = (ReindexOp*)(bop->y->input());
            LOGvvvv << "conv like op" << fop << fop->get_jit_key();

            OpInspector oi1(riop1);
            OpInspector oi2(riop2);


            if (oi1.mm[0] == 0 && oi2.mm[0] == 0) {
                // dw
                // x.mm [0,1,-1,1,2,3,2,3] y.mm [0,1,1,-1,2,3,-1,-1] w.mm [-1,0,0,1,-1,-1,2,3]
                OpInspector xoi(oi1.mm[2] == -1 ? riop1 : riop2);
                OpInspector yoi(oi1.mm[2] == -1 ? riop2 : riop1);
                OpInspector woi(rop);
                int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
                int zn, zg, zci, zco, zh, zw, zwh, zww;
                zn = zg = zci = zco = zh = zw = zwh = zww = 0;
                xoi.get_id(xoi.m1 & woi.m2 & yoi.m1, zn);
                xoi.get_id(xoi.m3 & woi.m3 & yoi.m3, zg);
                xoi.get_id(xoi.m3 & woi.m2 & yoi.m1, zh, zw);
                xoi.get_id(xoi.m2 & woi.m3 & yoi.m3, zco);
                xoi.get_id(xoi.m3 & woi.m1 & yoi.m2, zci, zwh, zww);
                LOGvvvv << "group conv backward dw zn,zg,zci,zco,zh,zw,zwh,zww =" << vector<int>{zn,zg,zci,zco,zh,zw,zwh,zww};
                xoi.check_overlap({zn,zg,zci,zco,zh,zw,zwh,zww});
                if (xoi.failed) continue;
                xn = xoi.mm[zn];
                xc = xoi.mm[zci];
                xh = xoi.mm[zh];
                xw = xoi.mm[zw];
                xformat = xoi.format("abcd", {xn, xc, xh, xw});
                wci = woi.mm[zci];
                wco = woi.mm[zco];
                wh = woi.mm[zwh];
                ww = woi.mm[zww];
                wformat = xoi.format("iohw", {wci, wco, wh, ww});
                yn = yoi.mm[zn];
                yc = yoi.mm[zco];
                yh = yoi.mm[zh];
                yw = yoi.mm[zw];
                yformat = xoi.format("abcd", {yn, yc, yh, yw});

                // mkl doesn't support "cdab" format
                if (yformat == "cdab") continue;
                // cuda doesn't support "iohw" format
                if (fop->flags.get(NodeFlags::_cuda) && wformat == "iohw") continue;
                if (xoi.failed) continue;
                
                std::stringstream ss;
                // i@zh*stride+i@zwh+padding
                ss << "i" << zh << "*stride+i" << zwh << "*dilation+padding";
                auto expr_h = expr::make(ss.str());

                vector<unique_ptr<Expr>> rh;
                auto src_h = expr::make(riop1->indexes[xh]);
                if (!expr::match(src_h.get(), expr_h.get(), {"stride", "padding", "dilation"}, {"i"+S(zh), "i"+S(zwh)}, rh)) {
                    LOGvvvv << "Expr not match" << src_h << expr_h;
                    continue;
                }
                if (!rh[0]->is(expr::_number) || !rh[1]->is(expr::_number)) continue;

                dw = rop->x;
                stride = rh[0]->as_int();
                padding = -rh[1]->as_int();
                dilation = rh[2]->as_int();
                kernel_size = dw->shape[wformat.find("h")];
                groups = (oi1.mm[2] == -1 ? riop1 : riop2)->x->shape[xc] / dw->shape[wci];

                if (fop->flags.get(NodeFlags::_cpu) && groups > 1) {
                    LOGi << "group conv does not support mkl";
                    continue;
                } 

                LOGvvvv << stride << padding << dilation << kernel_size << groups;

                x = (oi1.mm[2] == -1 ? riop1 : riop2)->x;
                y = (oi1.mm[2] == -1 ? riop2 : riop1)->x;
                bo++;
            } else {
                // dx
                OpInspector woi(oi1.mm[0] == -1 ? riop1 : riop2);
                OpInspector yoi(oi1.mm[0] == -1 ? riop2 : riop1);
                OpInspector xoi(rop);
                int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
                int zn, zg, zci, zco, zh, zw, zwh, zww;
                zn = zg = zci = zco = zh = zw = zwh = zww = 0;
                xoi.get_id(xoi.m1 & woi.m2 & yoi.m1, zn);
                xoi.get_id(xoi.m3 & woi.m3 & yoi.m3, zg);
                xoi.get_id(xoi.m3 & woi.m2 & yoi.m1, zh, zw);
                xoi.get_id(xoi.m2 & woi.m3 & yoi.m3, zco);
                xoi.get_id(xoi.m3 & woi.m1 & yoi.m2, zci, zwh, zww);
                LOGvvvv << "group conv backward dx zn,zg,zci,zco,zh,zw,zwh,zww =" << vector<int>{zn,zg,zci,zco,zh,zw,zwh,zww};
                xoi.check_overlap({zn,zg,zci,zco,zh,zw,zwh,zww});
                if (xoi.failed) continue;
                xn = xoi.mm[zn];
                xc = xoi.mm[zci];
                xh = xoi.mm[zh];
                xw = xoi.mm[zw];
                xformat = xoi.format("abcd", {xn, xc, xh, xw});
                wci = woi.mm[zci];
                wco = woi.mm[zco];
                wh = woi.mm[zwh];
                ww = woi.mm[zww];
                wformat = xoi.format("iohw", {wci, wco, wh, ww});
                yn = yoi.mm[zn];
                yc = yoi.mm[zco];
                yh = yoi.mm[zh];
                yw = yoi.mm[zw];
                yformat = xoi.format("abcd", {yn, yc, yh, yw});
                // mkl doesn't support "cdab" format
                if (yformat == "cdab") continue;
                // cuda doesn't support "iohw" format
                if (fop->flags.get(NodeFlags::_cuda) && wformat == "iohw") continue;
                if (xoi.failed) continue;
                
                std::stringstream ss;
                // i@zh*stride+i@zwh+padding
                ss << "i" << zh << "*stride+i" << zwh << "*dilation+padding";
                auto expr_h = expr::make(ss.str());

                vector<unique_ptr<Expr>> rh;
                auto src_h = expr::make(rop->indexes[xh]);
                if (!expr::match(src_h.get(), expr_h.get(), {"stride", "padding", "dilation"}, {"i"+S(zh), "i"+S(zwh)}, rh)) {
                    LOGvvvv << "Expr not match" << src_h << expr_h;
                    continue;
                }
                if (!rh[0]->is(expr::_number) || !rh[1]->is(expr::_number)) continue;
                
                dx = rop->x;         
                stride = rh[0]->as_int();
                padding = -rh[1]->as_int();
                dilation = rh[2]->as_int();
                height = dx->shape[xformat.find("c")];
                width = dx->shape[xformat.find("d")];
                groups = dx->shape[xc] / (oi1.mm[0] == -1 ? riop1 : riop2)->x->shape[wci];

                if (fop->flags.get(NodeFlags::_cpu) && groups > 1) {
                    LOGi << "group conv does not support mkl";
                    continue;
                } 

                LOGvvvv << stride << padding << dilation << height << width << groups;
                
                w = (oi1.mm[0] == -1 ? riop1 : riop2)->x;
                y = (oi1.mm[0] == -1 ? riop2 : riop1)->x;
                bo+=2;
            }

        }

        // TODO: CUDA only support nchw(abcd)
        if (fop->flags.get(NodeFlags::_cuda) && (xformat != "abcd" || yformat != "abcd"))
            continue;

        if (bo&1) {
            auto make_conv_w = get_op_info(
                    "cudnn_conv_backward_w"
                ).get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, string, string, string>();
            auto rvar_w = make_conv_w(x, y, kernel_size, stride, padding, dilation, groups, xformat, wformat, yformat);
            auto rid = fop->context->vrm.add_relay_group({{rvar_w, dw}});
            if (rid>=0) {
                auto srid = "relay"+S(rid);
                add_candidate(srid, 1);
                add_candidate(srid, 0);
                confidence = 20;
            }
        }
        if (bo&2) {
            auto make_conv_x = get_op_info(
                    "cudnn_conv_backward_x"
                ).get_constructor<VarPtr, Var*, Var*, int , int, int, int, int, int, string, string, string>();
            auto rvar_x = make_conv_x(w, y, height, width, stride, padding, dilation, groups, xformat, wformat, yformat);
            auto rid = fop->context->vrm.add_relay_group({{rvar_x, dx}});
            if (rid>=0) {
                auto srid = "relay"+S(rid);
                add_candidate(srid, 1);
                add_candidate(srid, 0);
                confidence = 20;
            }
        }
    }
}

void GroupConvTuner::run(PassManager* pm, TunerManager* tm) {
    FusedOp* fop=tm->oc->op;

    forwardTune(fop);
    backwardTune(fop);
}

}
