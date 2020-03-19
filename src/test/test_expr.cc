// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "opt/expr.h"

namespace jittor {

using namespace expr;

JIT_TEST(expr) {
    auto check = [&](const string& src, const string& exp, int debug=1, int try_reduce_braces=0) {
        LOGv << "test" << src << exp;
        unique_ptr<Expr> expr(new Expr(src));
        ASSERTop(expr->to_string(try_reduce_braces, debug),==,exp);

        string nexp = expr->to_string(try_reduce_braces);
        expr.reset(new Expr(nexp));
        ASSERTop(expr->to_string(try_reduce_braces),==,nexp);
        if (try_reduce_braces) return;

        string exp2 = expr->to_string(1);
        expr.reset(new Expr(exp2));
        ASSERTop(expr->to_string(0),==,nexp) << "Test reduce braces failed" << "\n" 
            << src << "\n" << exp2;
    };
    auto check_error = [&](const string& src) {
        expect_error([&]() {
            unique_ptr<Expr> expr(new Expr(src));
        });
    };
    check("asd", "asd");
    check("a+b", "(/*f:is_binary_op,is_op,;s:+;c:2*/a+b)");
    check("a+b*c", "(/*f:is_binary_op,is_op,;s:+;c:2*/a+(/*f:is_binary_op,is_op,;s:*;c:2*/b*c))");
    check("a * (b+c )/d", "((a*(b+c))/d)", 0);
    check("-a * b", "(-(a*b))", 0);
    check("a+(b+c)", "(a+b+c)", 0);
    check("a(b+c)", "(a((b+c)))", 0);
    check("a[b+c]", "(a[(b+c)])", 0);
    check("a{b+c}", "(a{(b+c)})", 0);
    check_error("a{b+c)");
    check_error("a(b+c)a");
    check("a+x(b+c)", "(a+(x((b+c))))", 0);
    check("a::x(b+c)", "((a::x)((b+c)))", 0);
    check("a+b? c & d: x && y", "((a+b)?(c&d):(x&&y))", 0);
    check("a=b?x:y", "(a=(b?x:y))", 0);
    check_error("a=b?x,c:y");
    check("a=b?c:y,b=c;d+=p", "((a=(b?c:y)),((b=c);(d+=p)))", 0);
    check("a,b+=c", "(a,(b+=c))", 0);
    check("a(x,y,z)", "(a(x,y,z))", 0);
    check("(a+b){x,y,z}+k", "(((a+b){x,y,z})+k)", 0);
    check("++a_b", "(++a_b)", 0);
    check("++a++", "((++a)++)", 0);
    check("*a", "(*a)", 0);
    check("a*", "(a*)", 0);
    check("a***", "(((a*)*)*)", 0);
    check_error("*");
    check("a***", "a***", 0, 1);
    // this test can not pass
    // check("***a", "***a", 0, 1);
    check("a((x),(y,z))", "a(x,(y,z))", 0, 1);
}

JIT_TEST(expr_get_tokens) {
    auto check = [&](
        const string& src, 
        const vector<pair<int,int>>& tokens,
        const vector<size_t>& flags
    ) {
        vector<pair<int,int>> t;
        vector<size_t> f;
        get_tokens(src, t, f);
        ASSERTop(t,==,tokens);
        ASSERTop(f,==,flags);
    };
    check("a+b",{{0,1},{1,2},{2,3}},{0, _op, 0});
    check(" a + b ",{{1,2},{3,4},{5,6}},{0, _op, 0});
    check("'a'",{{0,3}},{_char});
    check("\"asdasd\"",{{0,8}},{_string});
    check("1 0x1a 0b1 1u 1ull",
        {{0,1},{2,6},{7,10},{11,13},{14,18}},
        {_int,_int,_int,_int,_int});
    check("1. 1.0f 1e3",
        {{0,2},{3,7},{8,11}},
        {_float,_float,_float});
    auto a = std::make_unique<Expr>("0xaa");
    ASSERTop(a->as_int(),==,0xaa);
    a = std::make_unique<Expr>("0b11");
    ASSERT(a->as_int()==0b11);
    a = std::make_unique<Expr>("123");
    ASSERT(a->as_int()==123);
    a = std::make_unique<Expr>("1.5");
    ASSERT(a->as_float()==1.5);
    a = std::make_unique<Expr>("2.");
    ASSERT(a->as_float()==2.);
    a = std::make_unique<Expr>("1e2");
    ASSERTop(a->as_float(),==,1e2);
}

JIT_TEST(expr_simplify) {
    auto check = [&](const string& a, const string& b) {
        LOGv << "test" << a << b;
        auto x = std::make_unique<Expr>(a);
        LOGv << *x;
        x = x->simplify();
        ASSERTop(x->to_string(1),==,b);
    };
    check("1+1","2");
    check("1+1*3+1.5","5.5");
    check("0?1+2:3+4","7");
    check("100/2*a", "50*a");
    check("100/2*a + 1/3.0*b", "50*a+0.333333*b");
    check("1+a+1+1+b+(1+c+1)+1+d+1+1","1+a+2+b+1+c+2+d+2");
    check("1*a*1", "a");
    check("a/1", "a");
    check("1/a", "1/a");
    check("a+0", "a");
    check("a*0", "0");
    // TODO: pass this test
    // check("a+1-1", "a");

    check("0+0+0", "0");
    check("(((0+(loop_cnt*1))*2)-0)","loop_cnt*2");
}

JIT_TEST(expr_expand) {
    auto check = [&](const string& a, const string& b) {
        LOGv << "test" << a << b;
        auto x = std::make_unique<Expr>(a);
        x = expand(x.get());
        ASSERTop(x->to_string(1),==,b);
    };
    check("-a", "(-1)*a");
    check("a-b", "a+(-1)*b");
    check("(a+b)*c", "a*c+b*c");
    check("(a-b)*c", "a*c+(-1)*b*c");
    check("(a-b)*(c-d)", "a*c+a*(-1)*d+(-1)*b*c+(-1)*b*(-1)*d");
    check("a&&b", "a&&b");
    check("!(a&&b)", "!a||!b");
    check("!(a&&b&&c&&d)", "!a||!b||!c||!d");
    check("!(a||b||c||d)", "!a&&!b&&!c&&!d");
    check("!!a", "a");
    check("!!!!a", "a");
    check("!!!a", "!a");
    check("!(!a&&b)", "a||!b");
    check("!(a>b && c<=d && e!=f)", "a<=b||c>d||e>=f&&e<=f");
    check("a@>b", "!a||b");
    check("a@<b", "!b||a");
    check("(a||b)&&c", "a&&c||b&&c");
    check("a==b", "a>=b&&a<=b");
    check("!(a!=b)", "a>=b&&a<=b");
    check("0<=i0 && i0<n && 0<=i1 && i1<n && 0<=j1 && j1<m && 0<=j1 && j1<m\
        && n>0 && m>0 \
        && (i0!=i1) @> (i0*m+j0 != i1*m+j0)",
    "0>i0||i0>=n||0>i1||i1>=n||0>j1||j1>=m||0>j1||j1>=m||n<=0||m<=0||i0>=i1&&i0<=i1||i0*m+j0<i1*m+j0||i0*m+j0>i1*m+j0");
    // vector<unique_ptr<string>> v{std::make_unique<string>("asd"), std::make_unique<string>("asxd")};
    // check("-a", "-1*a");
    // for i in n:
    //     for j in m
    //         write(i*m+j)
    // 0<=i0 && i0<n && 0<=i1 && i1<n && 0<=j1 && j1<m && 0<=j1 && j1<m
    // && n>0 && m>0
    // && (i0<i1 || i0>i1) -> (i0*m+j0 < i1*m+j0 || i0*m+j0 > i1*m+j0)

    // for i in n:
    //     for j in m:
    //         (i*m+j)*a+b
    // get_coeff([i,j], &coeff, &b)
    // match(m*i+j+k, a*i+b*j+c, [a,b,c], [i,j], results) -> bool 
}

JIT_TEST(expr_match) {
    auto check = [&](string src, string target, 
        vector<string> solve_symbols,
        vector<string> exclude_symbols,
        vector<string> results,
        bool ret = true
    ) {
        auto _src = make(src);
        auto _target = make(target);
        LOGv << "test" << src << target;
        vector<unique_ptr<Expr>> _results;
        bool _ret = match(_src.get(), _target.get(), solve_symbols, exclude_symbols, _results);
        ASSERT(ret==_ret) << src << target;
        if (ret) {
            ASSERT(results.size()==_results.size());
            for (uint i=0; i<results.size(); i++)
                ASSERTop(results[i],==,_results[i]->to_string(1));
        }
    };
    check("1", "a", {"a"}, {}, {"1"});
    check("1+x", "a", {"a"}, {}, {"1+x"});
    check("y", "x", {}, {}, {}, false);
    check("x", "x", {}, {}, {}, true);
    check("1-2+x", "a", {"a"}, {}, {"(-1)+x"});
    check("3*i+j-1", "stride*i+j+pad", {"stride", "pad"}, {"i", "j"}, {"3", "(-1)"});
    check("i*2+j*2", "(i+j)*2", {}, {}, {});
    check("i*2+j*2", "(i+j)*a", {"a"}, {}, {"2"});
    check("i*2+j*3", "(i+j)*a", {"a"}, {}, {"2"}, false);
    check("3*i+j-1", "i*stride+pad+j", {"stride", "pad"}, {"i", "j"}, {"3", "(-1)"});
    check("1*i+j-1", "i*stride+pad+j", {"stride", "pad"}, {"i", "j"}, {"1", "(-1)"});
    check("1*i+j-1", "i*stride*stride+pad+j", {"stride", "pad"}, {"i", "j"}, {"1", "(-1)"});
    check("1*i+j", "i*stride*stride+pad+j", {"stride", "pad"}, {"i", "j"}, {"1", "0"});
}

} // jittor
