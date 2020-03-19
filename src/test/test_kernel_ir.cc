// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "opt/kernel_ir.h"

namespace jittor {

string str_diff_detail(const string& a, const string& b) {
    if (a.size() != b.size()) {
        return "size not match " + S(a.size()) + " vs " + S(b.size());
    }
    for (int i=0; i<a.size(); i++) {
        if (a[i] != b[i]) {
            string msg;
            msg = "a["+S(i)+"] != b["+S(i)+"] ("+a[i]+" vs "+b[i]+")\n";
            return msg;
        }
    }
    return "";
}

#define CHECK_STR(a,b) CHECK(a==b) << "\n----a----\n" << a << "\n----b----\n" << b \
    << "\n----diff----\n" \
    << str_diff_detail(a, b)

JIT_TEST(kernel_ir) {
    KernelIR ir(R"(
        #include <cmath>
        #define aaa  bbb
        using namespace std;
        void main(float* __restrict__ c) {
            // test commment
            int n = 1024;
            int m = 1024;
            int k = 1024;
            float* __restrict__ a = new float[n*m];
            float* __restrict__ b = new float[m*k];
            for (int i=0; i<n; i++)
                for (int j=0; j<k; j++)
                    c[i*k+j] = 0;
            for (int i=0; i<n; i++)
                for (int j=0; j<m; j++)
                    for (int l=0; l<k; l++)
                        c[i*k+j] += a[i*m+j] * b[j*k+l];
            if (c[0]==0)
                cout << "hahaha" << endl;
        })", true
    );
    string code = R"(// 
// scope: main(1), 

// C macro code:"#include <cmath>"
#include <cmath>
// C macro code:"#define aaa  bbb"
#define aaa  bbb
// C  code:"using namespace std;" raw:"1"
using namespace std;
// C func dtype:"void" lvalue:"main"
// scope: a(1), b(1), c(1), k(1), m(1), n(1), 
void main(float* __restrict__ c) {
    // C comment code:"// test commment"
    // test commment
    // C define dtype:"int" lvalue:"n" raw:"1" rvalue:"1024"
    int n = 1024;
    // C define dtype:"int" lvalue:"m" raw:"1" rvalue:"1024"
    int m = 1024;
    // C define dtype:"int" lvalue:"k" raw:"1" rvalue:"1024"
    int k = 1024;
    // C define dtype:"float* __restrict__" lvalue:"a" raw:"1" rvalue:"new float[n*m]"
    float* __restrict__ a = new float[n*m];
    // C define dtype:"float* __restrict__" lvalue:"b" raw:"1" rvalue:"new float[m*k]"
    float* __restrict__ b = new float[m*k];
    // C loop raw:"1"
    // scope: i(1), 
    for (int i = 0; i<n; i++) {
        // C loop raw:"1"
        // scope: j(1), 
        for (int j = 0; j<k; j++) {
            // C  code:"c[i*k+j] = 0;" raw:"1"
            c[i*k+j] = 0;
        }
    }
    // C loop raw:"1"
    // scope: i(1), 
    for (int i = 0; i<n; i++) {
        // C loop raw:"1"
        // scope: j(1), 
        for (int j = 0; j<m; j++) {
            // C loop raw:"1"
            // scope: l(1), 
            for (int l = 0; l<k; l++) {
                // C  code:"c[i*k+j] += a[i*m+j] * b[j*k+l];" raw:"1"
                c[i*k+j] += a[i*m+j] * b[j*k+l];
            }
        }
    }
    // C if
    if (c[0]==0) {
        // C  code:"cout << "hahaha" << endl;" raw:"1"
        cout << "hahaha" << endl;
    }
}
)";
    string code2 = ir.to_string(0, true);
    CHECK_STR(code2, code);
}

JIT_TEST(kernel_ir_utils) {
    CHECK(startswith("abcd", "ab"));
    CHECK(!startswith("abcd", "b"));
    CHECK(!startswith("abcd", "ab", 0, true));
    CHECK(startswith("abcd", "abcd", 0, true));
    CHECK(startswith("abcd", "cd", 2, true));
    CHECK(startswith("abcd", "cd", 2, true, 4));
    CHECK(!startswith("abcd", "cd", 2, true, 3));
    CHECK(startswith("abcd", "c", 2, true, 3));

    CHECK(endswith("abcd", "bcd"));
    CHECK(!endswith("abcd", "bc"));

    CHECK(split("a,b,c", ",") == vector<string>({"a","b","c"}));
    CHECK(split("a,b,c,,", ",") == vector<string>({"a","b","c","",""}));
    CHECK(split("a,b,c,,", ",", 3) == vector<string>({"a","b","c,,"}));
}

JIT_TEST(kernel_ir_manip) {
    KernelIR ir(R"(for (int i=0; i<n; i++) a[i]=0;)");
    auto& c = ir.children;
    ir.push_front("a[0]++;");
    ir.push_back("a[0]--;");
    CHECKop(c.size(),==,3);
    ir.push_back("a[1]++;", &ir.before);
    CHECKop(ir.before.size(),==,1);
    ir.push_back("a[2]++;", &ir.after);
    CHECKop(ir.after.size(),==,1);
    ir.push_back("a[2]++;", &ir.inner);
    CHECKop(ir.inner.size(),==,4);
    ir.insert(1, "  a[2]++;");
    CHECKop(c.size(),==,4);
    ir.push_back("auto* xxx xxx xxx;");
    string code = R"(// B  code:"a[1]++;"
a[1]++;
// loop dtype:"int" lvalue:"i" rvalue:"n"
// scope: i(1), xxx(1), 
for (int i = 0; i<n; i++) {
    // I  code:"a[2]++;"
    a[2]++;
    // C  code:"a[0]++;"
    a[0]++;
    // C  code:"a[2]++;"
    a[2]++;
    // C  code:"a[i]=0;"
    a[i]=0;
    // C  code:"a[0]--;"
    a[0]--;
    // C define dtype:"auto* xxx xxx" lvalue:"xxx"
    auto* xxx xxx xxx;
}
// A  code:"a[2]++;"
a[2]++;
)";
    CHECK_STR(ir.to_string(0, 1), code);
    c[0]->erase();
    c[0]->erase();
    c.back()->erase();
    c.back()->erase();
    CHECKop(c.size(),==,1);
    ir.push_back(ir.clone());
    CHECK(c.back()->type == "loop");
    auto& in_loop = *c.back();
    // test replace
    in_loop.replace({{"i", "j"}, {"n", "range0"}}, true, false);
    // test swap
    ir.swap(in_loop);
    CHECK(ir.attrs["lvalue"]=="j");
    in_loop.replace({{"n", "range1"}}, true, false);
    // test rename_loop_index
    ir.rename_loop_index();
    CHECK(ir.attrs["lvalue"]=="id0");
    CHECK(in_loop.attrs["lvalue"]=="id1");
    // test find_loops
    auto a = ir.find_loops("1");
    CHECK(a.size()==1 && a[0]==&in_loop);
    // test find_define
    auto* b = in_loop.find_define("id0");
    CHECK(b == ir.inner[0].get()) << b;
    // test move_loop_back
    ir.push_back("a[3]++;");
    ir.push_back("for (int i=0; i<n; i++) a[i]=i;", 0, true);
    ir.move_loop_back();
    CHECK(c.back().get()==&in_loop);
    // test merge_loop
    ir.move_out_children();
    ir.push_back("for (int i=0; i<range3; i++) for (int i=0; i<range4; i++) a[i]=i;");
    ir.push_back("for (int i=0; i<range3; i++) for (int i=0; i<range4; i++) a[i]++;");
    ir.merge_loop();
    CHECK(c.size()==1);
    CHECK(c[0]->children.size()==1);
    CHECK(c[0]->children[0]->children.size()==2);
    // test expand_empty_block
    ir.move_out_children();
    ir.push_back("{ T xx=1; xx++; a[xx]=0; }");
    CHECK(ir.scope.count("xx")==0 && c.back()->scope.count("xx"));
    ir.expand_empty_block();
    CHECK(c.size()==3 && ir.scope.count("xx"));
    // test solve_conflict_define
    ir.move_out_children();
    ir.push_back("{ T xx=1; xx++; a[xx]=0; }");
    ir.push_back("{ T xx=1; xx++; a[xx]=0; }");
    ir.expand_empty_block();
    ir.solve_conflict_define();
    CHECK(c.size()==6 &&
        c[0]->attrs["lvalue"] == "xx" &&
        c[1]->attrs["code"] == "xx++;" &&
        c[2]->attrs["code"] == "a[xx]=0;" &&
        c[3]->attrs["lvalue"] == "xx_" &&
        c[4]->attrs["code"] == "xx_++;" &&
        c[5]->attrs["code"] == "a[xx_]=0;"
    );
    // test remove_unused
    // a <-+- c <-- d (unused)
    //     +-- b (used)
    ir.move_out_children();
    ir.push_back("T a=0;");
    ir.push_back("T b=a;");
    ir.push_back("T c=a;");
    ir.push_back("b++;");
    ir.push_back("T d=c;");
    ir.check_unused();
    CHECK(c.size()==5 &&
        c[0]->check_attr("used", "1") &&
        c[1]->check_attr("used", "1") &&
        c[2]->check_attr("used", "1") &&
        !c[4]->check_attr("used", "1")
    );
    ir.remove_all_unused();
    CHECK(c.size()==3);
    // test split_loop 1
    ir.move_out_children();
    ir.push_back("for (int i=0; i<range1; i++) a[i]=i;");
    expect_error([&]() {ir.split_loop(1, 2);});
    // test split_loop 2
    ir.move_out_children();
    ir.push_back("for (int i=0; i<range1; i++) a[i]=i;");
    ir.rename_loop_index();
    ir.push_front("int range1=1024;");
    ir.push_front("int stride1=128;");
    ir.split_loop(1, 2);
    code = R"(for (int id1 = 0; id1<range1; id1+=stride1) {
    int range2 = 128;
    for (int id2 = 0; id2<range2; id2++) {
        a[(id1+id2)]=(id1+id2);
    }
}
)";
    CHECK(c.back()->to_string() == code);
    // test split_loop 3
    ir.move_out_children();
    ir.push_back("for (int i=0; i<range1; i++) a[i]=i;");
    ir.rename_loop_index();
    ir.push_front("int range1=1024;");
    ir.push_front("int stride1=xx;");
    ir.split_loop(1, 2);
    code = R"(for (int id1 = 0; id1<range1; id1+=stride1) {
    int range2 = ::min(range1-id1, stride1);
    for (int id2 = 0; id2<range2; id2++) {
        a[(id1+id2)]=(id1+id2);
    }
}
)";
    CHECKop(c.back()->to_string(),==,code);
    // test get_number
    ir.move_out_children();
    ir.push_back("T x=1;");
    ir.push_back("T y=n;");
    int num;
    CHECK(ir.get_number("x", num) && num==1);
    CHECK(!ir.get_number("z", num) && num==-1);
    CHECK(!ir.get_number("y", num) && num==-2);
    // test resplit
    ir.move_out_children();
    ir.push_back("for (int i=0; i<range1; i++) a[i]=i;");
    ir.rename_loop_index();
    ir.push_front("int range1=1024;");
    ir.push_front("int stride1=xx;");
    ir.split_loop(1, 2);
    c.back()->resplit();
    code = R"(int id1 = 0;
for (id1=0; id1+stride1<=range1; id1+=stride1) {
    int range2 = stride1;
    for (int id2 = 0; id2<range2; id2++) {
        a[(id1+id2)]=(id1+id2);
    }
}
if (id1<range1) {
    int range2 = range1-id1;
    for (int id2 = 0; id2<range2; id2++) {
        a[(id1+id2)]=(id1+id2);
    }
}
)";
    CHECKop(c.back()->to_string(),==,code);
}

JIT_TEST(kernel_ir_func) {
    KernelIR ir("");
    ir.push_back("void func1() {func0(0, 1);}");
    auto func1 = ir.children.back().get();
    func1->push_back("void func0(int a, int b) {}", &func1->before);
    auto func0 = func1->before.back().get();
    CHECK(func0->inner.size()==2);
    ir.remove_all_unused();
    CHECK(func0->inner.size()==0);
    CHECK(func1->children.back()->get_attr("code") == "func0();");
    // test remove_func_call_arg
    string s = "func(0,1,2,(1,2),3);";
    expect_error([&]() {remove_func_call_arg(s, 5);});
    remove_func_call_arg(s, 4);
    CHECKop(s,==,"func(0,1,2,(1,2));");
    remove_func_call_arg(s, 2);
    CHECKop(s,==,"func(0,1,(1,2));");
    remove_func_call_arg(s, 2);
    CHECKop(s,==,"func(0,1);");
    remove_func_call_arg(s, 0);
    CHECKop(s,==,"func(1);");
    remove_func_call_arg(s, 0);
    CHECKop(s,==,"func();");
}

JIT_TEST(kernel_ir_swap_scope) {
    KernelIR ir(R"(
        void func() {
            for (int i=0; i<n; i++)
                for (int j=0; j<n; j++)
                    k = i+j;
        }
    )");
    auto& loop1 = ir.children.back();
    auto& loop2 = loop1->children.back();
    loop1->swap(*loop2);
    CHECK(loop1->scope.count("j"));
    CHECK(loop2->scope.count("i"));
    CHECK(loop1->scope["j"].size()==1);
    CHECK(loop2->scope["i"].size()==1);
}

JIT_TEST(kernel_ir_remove_intermediate) {
    KernelIR ir(R"(
        void func() {
            int* xp = input;
            int* yp = output;
            for (int i=0; i<100; i++) {
                yp[i] = xp[i]+1;
            }
        }
    )");
    ir.remove_intermediate({"y"});
    string expect = "auto yd = xp[i]+1;\n";
    CHECK(ir.children.at(1)->children.at(0)->to_string()==expect);
    ir.remove_all_unused();
    CHECK(ir.children.at(0)->children.size()==0);
}

} // jittor