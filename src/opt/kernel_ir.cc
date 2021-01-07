// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include "opt/kernel_ir.h"

namespace jittor {

template<class T>
vector<typename unordered_map<string,T>::iterator> sort(unordered_map<string,T>& m) {
    vector<typename unordered_map<string,T>::iterator> v;
    v.reserve(m.size());
    for (auto i=m.begin(); i!=m.end(); ++i)
        v.push_back(i);
    auto cmp = [](const auto& a, const auto& b) -> bool {
        return a->first < b->first;
    };
    sort(v.begin(), v.end(), cmp);
    return v;
}

bool isvar(char x) { return isalnum(x) || x == '_' || x == ':'; }

std::ostream& operator<<(std::ostream& os, KernelIR& ir) {
    return os << ir.to_string();
}

bool startswith(const string& a, const string& b, uint start, bool equal, uint end) {
    if (!end) end = a.size();
    if (b.size()+start > end) return false;
    if (equal && b.size()+start != end) return false;
    for (uint i=0; i<b.size(); i++)
        if (a[i+start] != b[i]) return false;
    return true;
}

bool endswith(const string& a, const string& b) {
    if (a.size() < b.size()) return false;
    return startswith(a, b, a.size()-b.size());
}

vector<string> split(const string& s, const string& sep, int max_split) {
    vector<string> ret;
    int pos = -1, pos_next;
    while (1) {
        pos_next = s.find(sep, pos+1);
        if (pos_next == (int)string::npos || (int)ret.size() == max_split-1) {
            ret.push_back(s.substr(pos+sep.size()));
            return ret;
        }
        ret.push_back(s.substr(pos+sep.size(), pos_next-pos-sep.size()));
        pos = pos_next;
    }
    ASSERT(max_split==0);
    return ret;
}

string strip(const string& s) {
    int i=0;
    while (i<s.size() && (s[i]==' ' || s[i]=='\t' || s[i]=='\n')) i++;
    int j = s.size();
    while (j>i && (s[j]==' ' || s[j]=='\t' || s[j]=='\n')) j--;
    return s.substr(i,j-i);
}

void KernelIR::del_scope() {
    if (father && (type=="define" || type=="func" || type=="macro")) {
        father->scope[attrs["lvalue"]].remove(this);
    }
}

void KernelIR::add_scope() {
    if (father && (type=="define" || type=="func" || type=="macro"))
        father->scope[get_attr("lvalue")].push_back(this);
}

void KernelIR::clear() {
    del_scope();
    type.clear();
    attrs.clear();
    for (int i=(int)inner.size()-1; i>=0; i--)
        inner[i]->erase();
}

string& KernelIR::get_attr(const string& s) {
    return attrs[s];
}

bool KernelIR::has_attr(const string& s) {
    auto iter = attrs.find(s);
    if (iter == attrs.end() || iter->second.size()==0)
        return false;
    return true;
}

void KernelIR::try_parse_define(const string& s) {
    // dtype lvalue = rvalue;
    clear();
    string& dtype = get_attr("dtype");
    string& lvalue = get_attr("lvalue");
    string& rvalue = get_attr("rvalue");
    int count=0;
    uint end=s.size();
    bool find_eq = 0;
    while (end && (s[end-1]==' ' || s[end-1]=='\n' || s[end-1]==';')) end--;
    for (uint i=0; i<end; i++) {
        while (i<end && (s[i]==' ' || s[i]=='\n')) i++;
        uint j=i+1;
        if (s[i]=='=' && s[i+1]!='=') {
            find_eq = 1;
            if (count<2) break;
            count++;
            continue;
        }
        if (i==end) break;
        while (j<end && !(s[j]==' ' || s[j]=='\n' || s[j]=='=')) j++;
        if (find_eq) {
            rvalue = s.substr(i,end-i);
            bool ok=true;
            for (char c : lvalue)
                if (!isvar(c)) ok=false;
            for (char c : dtype)
                if (!isvar(c) && c!=' ' && c!='*') ok=false;
            if (!ok)
                // this is not a definition
                break;
            type = "define";
            add_scope();
            return;
        }
        if (count==0)
            dtype = s.substr(i,j-i);
        else if (count==1)
            lvalue = s.substr(i,j-i);
        else {
            dtype += " ";
            dtype += lvalue;
            lvalue = s.substr(i,j-i);
        }
        i = j-1;
        count++;
    }
    if (!find_eq) {
        bool ok=1;
        // using namespace xxx; is not define
        if (startswith(dtype, "using ")) ok=0;
        // check is definition or not
        for (char c : dtype)
            if (!(isvar(c) || c==' ' || c=='*')) {
                ok=0;
                break;
            }
        if (ok) {
            type = "define";
            return;
        }
    }
    attrs = {{"code",s}};
}


void KernelIR::parse_for_loop(const string& s, bool raw) {
    // parse: for (dtype lvalue = 0; lvalue<rvalue; lvalue++)
    clear();
    if (!raw) {
        int count=0;
        string& dtype = get_attr("dtype");
        string& lvalue = get_attr("lvalue");
        string& rvalue = get_attr("rvalue");
        for (uint i=0; i<s.size(); i++) {
            while (i<s.size() && !isvar(s[i])) i++;
            uint j=i;
            while (j<s.size() && isvar(s[j])) j++;
            if (i==j) break;
            if (count==0)
                ASSERT(startswith(s, "for", i));
            else if (count==1)
                dtype = s.substr(i,j-i);
            else if (count==2)
                lvalue = s.substr(i,j-i);
            else if (count==3)
                ASSERT(startswith(s, "0", i));
            else if (count==4)
                ASSERT(startswith(s, lvalue, i));
            else if (count==5)
                rvalue = s.substr(i,j-i);
            else if (count==6)
                ASSERT(startswith(s, lvalue, i));
            i = j;
            count++;
        }
        ASSERTop(count,==,7) << "Parse for loop failed:" << s;
        if (startswith(rvalue, "range"))
            attrs["loop_id"] = rvalue.substr(5);
    } else
        attrs["raw"] = "1";
    type = "loop";
    // find '(' and ')', then split by ';'
    int l=0,r=s.size()-1;
    while (l<(int)s.size() && s[l]!='(') l++;
    while (r>=0 && s[r]!=')') r--;
    ASSERT(l<r);
    auto vs = split(s.substr(l+1, r-l-1), ";");
    ASSERTop(vs.size(),==,3) << s;
    for (auto& s : vs)
        push_back(s+";", &inner);
}

void KernelIR::push_back(const string& src, vector<unique_ptr<KernelIR>>* ls, bool raw) {
    if (!ls) ls = &children;
    ASSERT(ls>=&before && ls<=&after);
    ls->emplace_back(std::make_unique<KernelIR>(src, raw));
    auto& ir = *ls->back();
    ir.father = this;
    ir.flist = ls;
    ir.add_scope();
}

void KernelIR::push_front(const string& src, vector<unique_ptr<KernelIR>>* ls, bool raw) {
    if (!ls) ls = &children;
    ASSERT(ls>=&before && ls<=&after);
    ls->insert(ls->begin(), std::make_unique<KernelIR>(src, raw));
    auto& ir = *ls->front();
    ir.father = this;
    ir.flist = ls;
    ir.add_scope();
}

void KernelIR::push_back(unique_ptr<KernelIR>&& irp, vector<unique_ptr<KernelIR>>* ls) {
    ASSERT(irp->father==nullptr);
    if (!ls) ls = &children;
    ASSERT(ls>=&before && ls<=&after);
    ls->emplace_back(move(irp));
    auto& ir = *ls->back();
    ir.father = this;
    ir.flist = ls;
    ir.add_scope();
}

void KernelIR::push_front(unique_ptr<KernelIR>&& irp, vector<unique_ptr<KernelIR>>* ls) {
    ASSERT(irp->father==nullptr);
    if (!ls) ls = &children;
    ASSERT(ls>=&before && ls<=&after);
    ls->insert(ls->begin(), move(irp));
    auto& ir = *ls->front();
    ir.father = this;
    ir.flist = ls;
    ir.add_scope();
}

void remove_func_call_arg(string& src, int arg_i) {
    int presum=0, aid=-1, prev=0;
    for (int i=0; i<(int)src.size(); i++) {
        if (src[i]=='(') presum++;
        if (presum==1 && (src[i]=='(' || src[i]==',' || src[i]==')')) {
            if (arg_i == aid) {
                if (src[i]==',' && arg_i==0) i++;
                src.erase(prev, i-prev);
                return;
            }
            aid++;
            prev = i+(src[i]=='(');
        }
        if (src[i]==')') presum--;
    }
    LOGf << "Func call do not have enough argument" << arg_i << src;
}

void KernelIR::erase() {
    ASSERT(father && flist);
    // if is a function argument
    if (father->type=="func" && flist==&father->inner) {
        string& func_name = father->get_attr("lvalue");
        uint i=0;
        while (i<flist->size() && flist->at(i).get() != this) i++;
        ASSERT(i < flist->size());
        auto used = father->find_used();
        for (auto c : used) {
            string& code = c->get_attr("code");
            if (c->type=="" && startswith(code, func_name))
                remove_func_call_arg(code, i);
        }
    }
    del_scope();
    for (uint i=0; i<flist->size(); i++)
        if ((*flist)[i].get() == this) {
            flist->erase(flist->begin()+i);
            return;
        }
    ASSERT(0);
}

template <typename Func>
void KernelIR::for_each_rev(Func&& func) {
    vector<unique_ptr<KernelIR>>* ls[] = {&before, &inner, &children, &after};
    for (auto& l : ls) {
        for (int i=(int)l->size()-1; i>=0; i--)
            func((*l)[i]);
    }
}

KernelIR* KernelIR::find_define(const string& name) {
    auto iter = scope.find(name);
    if (iter == scope.end() || iter->second.size()==0) {
        if (father)
            return father->find_define(name);
        return nullptr;
    }
    ASSERT(iter->second.size()==1) << 
        "Name" << name << (iter->second.size()?"duplicate":"not found")
        << this->to_string(0,1) << scope;
    return iter->second.back();
}

unique_ptr<KernelIR> KernelIR::clone(bool with_children) {
    auto ir = std::make_unique<KernelIR>();
    ir->type = type;
    ir->attrs = attrs;
    for (auto& c : before)
        ir->push_back(c->clone(), &ir->before);
    for (auto& c : inner)
        ir->push_back(c->clone(), &ir->inner);
    if (with_children)
        for (auto& c : children)
            ir->push_back(c->clone(), &ir->children);
    for (auto& c : after)
        ir->push_back(c->clone(), &ir->after);
    return ir;
}

void KernelIR::rebuild_scope() {
    scope.clear();
    for_each([&](unique_ptr<KernelIR>& c) {
        c->add_scope();
    });
}

void KernelIR::update_father() {
    auto update = [&](vector<unique_ptr<KernelIR>>* flist) {
        for (auto& c : *flist)
            c->flist = flist, c->father = this;
    };
    update(&before);
    update(&inner);
    update(&children);
    update(&after);
}

void KernelIR::swap(KernelIR& other, bool with_children) {
    std::swap(type, other.type);
    std::swap(attrs, other.attrs);
    std::swap(before, other.before);
    std::swap(inner, other.inner);
    if (with_children) std::swap(children, other.children);
    std::swap(after, other.after);
    update_father();
    other.update_father();
    rebuild_scope();
    other.rebuild_scope();
}

unique_ptr<KernelIR> KernelIR::move_out() {
    ASSERT(father && flist);
    del_scope();
    int i=(int)flist->size()-1;
    for (; i>=0; i--)
        if ((*flist)[i].get() == this)
            break;
    ASSERT(i < (int)flist->size());
    unique_ptr<KernelIR> ir = move((*flist)[i]);
    flist->erase(flist->begin()+i);
    flist = nullptr;
    father = nullptr;
    return ir;
}

vector<unique_ptr<KernelIR>> KernelIR::move_out_children() {
    vector<unique_ptr<KernelIR>> cs(children.size());
    int i=(int)children.size()-1;
    for (; i>=0; i--) cs[i] = children[i]->move_out();
    return cs;
}

bool KernelIR::check_attr(const string& k, const string& v) {
    auto iter = attrs.find(k);
    return iter!= attrs.end() && iter->second==v;
}

vector<KernelIR*> KernelIR::find_loops(string lid) {
    vector<KernelIR*> q({this}), loops;
    for (uint i=0; i<q.size(); i++) {
        KernelIR* ir = q[i];
        if (ir->check_attr("loop_id", lid)) {
            loops.push_back(ir);
        }
        ir->for_each([&](unique_ptr<KernelIR>& c) {
            q.push_back(c.get());
        });
    }
    return loops;
}

string KernelIR::to_string(int level, bool debug) {
    if (level==0 && debug) {
        check_father();
    }
    if (level==0 && type=="" && children.size()) {
        level--;
    }
    std::stringstream s;
    //TODO: no level up for before & after
    //bool level_up = (before.size() || after.size()) && level>0;
    bool level_up = (before.size() || after.size()) && level>0 && (type != "define" && type != "");
    if (level_up) {
        for (int i=0; i<level*4; i++) s << ' ';
        s << "{\n";
        level++;
    }
    for (auto& c : before) s << c->to_string(level, debug);
    if (debug) {
        for (int i=0; i<level*4; i++) s << ' ';
        s << "// ";
        if (father) {
            if (flist == &father->children) s << "C";
            if (flist == &father->before) s << "B";
            if (flist == &father->inner) s << "I";
            if (flist == &father->after) s << "A";
            s << " ";
        }
        s << type;
        for (auto kv : sort(attrs))
            if (kv->second.size()) s << " " << kv->first << ":\"" << kv->second << '"';
        s << "\n";
        if (scope.size()) {
            for (int i=0; i<level*4; i++) s << ' ';
            s << "// scope: ";
            for (auto kv : sort(scope))
                s << kv->first << '(' << kv->second.size() << "), ";
            s << "\n";
        }
    }
    for (int i=0; i<level*4; i++) s << ' ';
    int inner_left=0;
    bool has_bc = false;
    if (type == "loop") {
        if (inner.size()) {
            ASSERT(inner.size()>=3);
            s << "for (";
            for (int i=0; i<3; i++) {
                auto c = inner[i]->to_string();
                c = c.substr(0, c.size()-2); // remove ;\n
                s << c << (i==2?"":"; ");
            }
            s << ") ";
            inner_left = 3;
            has_bc = true;
        } else {
            // empty loop
            has_bc = true;
        }
    } else if (type == "if") {
        ASSERT(inner.size()>=1);
        auto src = inner[0]->to_string();
        s << "if (" << src.substr(0, src.size()-2) << ") ";
        inner_left = 1;
        has_bc = true;
    } else if (type == "define") {
        s << attrs["dtype"] << " " << attrs["lvalue"];
        if (has_attr("rvalue"))
            s << " = " << attrs["rvalue"];
        s << ";\n";
    } else if (type == "func") {
        s << attrs["dtype"] << ' ' << attrs["lvalue"] << '(';
        for (uint i=0; i<inner.size(); i++) {
            if (i) s << ", ";
            auto arg = inner[i]->to_string();
            s << arg.substr(0, arg.size()-2);
            inner_left = inner.size();
        }
        s << ") ";
        has_bc = true;
    } else if (father) {
        auto iter = attrs.find("code");
        ASSERT(iter != attrs.end()) << attrs << type << father;
        s << iter->second << "\n";
        has_bc = attrs.count("has_bc");
    } else {
        s << "\n";
    }
    if (has_bc) s << "{\n";
    for (uint i=inner_left; i<inner.size(); i++)
        s << inner[i]->to_string(level+1, debug);
    for (auto& c : children)
        s << c->to_string(level+1, debug);
    if (has_bc) {
        for (int i=0; i<level*4; i++) s << ' ';
        s << "}\n";
    }
    
    for (auto& c : after) s << c->to_string(level, debug);
    if (level_up) {
        for (int i=0; i<level*4-4; i++) s << ' ';
        s << "}\n";
    }
    return s.str();
}
void KernelIR::insert(uint pos, const string& src, bool raw) {
    children.insert(children.begin()+pos, std::make_unique<KernelIR>(src, raw));
    auto& ir = *children[pos];
    ir.father = this;
    ir.flist = &children;
    ir.add_scope();
}

void KernelIR::insert(uint pos, vector<unique_ptr<KernelIR>>& irs) {
    vector<unique_ptr<KernelIR>> irs2(irs.size());
    for (int i=(int)irs.size()-1; i>=0; i--) {
        if (irs[i]->father)
            irs2[i] = irs[i]->move_out();
        else
            irs2[i] = move(irs[i]);
    }
    children.insert(
        children.begin()+pos,
        make_move_iterator(irs2.begin()),
        make_move_iterator(irs2.end())
    );
    for (uint i=0; i<irs2.size(); i++) {
        auto& c = children[i+pos];
        c->father = this;
        c->flist = &children;
        c->add_scope();
    }
}

void KernelIR::check_father() {
    for_each([&](unique_ptr<KernelIR>& c) {
        ASSERTop(c->father,==,this) << "father attrs:" << attrs << "attrs:" << c->attrs;
        c->check_father();
    });
}

bool KernelIR::get_number(const string& name, int& num) {
    auto iter = scope.find(name);
    if (iter == scope.end()) {
        if (father)
            return father->get_number(name, num);
        num = -1;
        return false;
    }
    ASSERT(iter->second.size()==1);
    auto snum = iter->second.back()->attrs["rvalue"];
    if (snum.size() && isdigit(snum[0])) {
        num = std::stoi(snum);
        return true;
    }
    num = -2;
    return false;
}

KernelIR::KernelIR(const string& src, bool raw) {
    uint end = src.size();
    uint start = 0;
    while (end && (src[end-1] == ' ' || src[end-1] == '\n')) end--;
    while (start<end && (src[start]==' ' || src[start]=='\n')) start++;
    type = "";
    for (uint i=start; i<end; i++) {
        int presum=0;
        uint j=i;
        while (j<end && (src[j]==' ' || src[j]=='\n')) j++;
        if (j+1<end && src[j]=='/' && src[j+1]=='/') {
            uint k=j+1;
            while (k<end && src[k]!='\n') k++;
            if (raw) {
                if (j==start && k==end) {
                    attrs["code"] = src;
                    type = "comment";
                    return;
                } else {
                    push_back(src.substr(j, k-j), nullptr, raw);
                }
            }
            i = k;
            continue;
        }
        if (src[j]=='#') {
            uint k=j+1;
            while (k<end && src[k]!='\n') k++;
            if (i==start && k==end) {
                attrs["code"] = src;
                type = "macro";
                auto v = split(src, " ", 3);
                ASSERT(v.size()>1);
                attrs["lvalue"] = v.at(1);
                attrs["rvalue"] = v.size()>2 ? v.at(2) : "";
                return;
            } else {
                push_back(src.substr(j, k-j), nullptr, raw);
                i = k;
                continue;
            }
        }
        if (j==end) return;
        uint k=j;
        while (k<end) {
            if (src[k] == '{' || src[k] == '(') presum++;
            if (src[k] == '}' || src[k] == ')') presum--;
            if (!presum && (src[k]==';' || src[k]=='}')) {
                presum = -1;
                k++;
                break;
            }
            k++;
        }
        ASSERT(presum == -1) << src << i << j << k << end;
        string s = src.substr(j, k-j);
        if (k==end && i==start) {
            if (startswith(s, "for ") || startswith(s, "if ")) {
                uint l = 0;
                while (l<s.size() && s[l]!=' ') l++;
                presum = 0;
                while (l < s.size()) {
                    if (s[l] == '(') presum++;
                    if (s[l] == ')') presum--;
                    if (presum==0 && s[l] == ')') break;
                    l++;
                }
                ASSERT(l<s.size() && presum==0) << s;
                if (startswith(s, "for "))
                    parse_for_loop(s.substr(0, l+1), raw);
                else {
                    type = "if";
                    // remove "if (" prefix
                    push_back(s.substr(4, l-4)+";", &inner, raw);
                }
                uint p = l+1;
                while (p<s.size() && (s[p]== ' ' || s[p]=='\n')) p++;
                if (p<s.size() && s[p] == '{') {
                    i = p;
                    end = k-1;
                    ASSERT(s[k-1] == '}');
                    continue;
                }
                i = l;
                continue;
            }
            if (s.size()>=2 && s[0]=='{' && s[s.size()-1]=='}') {
                // empty loop
                type = "loop";
                end--;
                continue;
            }
            // func define
            if (s.size()>=2 && s.back()=='}') {
                int l = s.find("{");
                ASSERT(l != string::npos);
                if (startswith(s, "namespace ")) {
                    // namespace xxx {...}
                    //               l
                    attrs["code"] = s.substr(0, l);
                    attrs["has_bc"] = "1";
                    type = "";
                    i = j + l;
                    end--;
                    continue;
                }
                int ll = s.rfind("(", l);
                int rr = s.rfind(")", l);
                // if () not found, maybe src like this:
                // vector<int> a = {...};
                if (ll<0 && rr<0) {
                    type = "";
                    attrs["code"] = src + ";";
                    return;
                }
                ASSERT(l>=0 && ll>=0 && rr>=0 && ll<rr && rr<l) << src;
                // dtype func_name(args...)  {  }
                //     y x        ll      rr l  end
                int x = ll;
                while (x>0 && s[x-1]!=' ') x--;
                int y = x-1;
                while (y>0 && s[y]==' ') y--;
                ASSERT(0<y && y<x && x<ll) << s << y << x << ll << rr << l;
                attrs["dtype"] = s.substr(0, y+1);
                attrs["lvalue"] = s.substr(x, ll-x);
                if (ll+1<rr) {
                    auto args = split(s.substr(ll+1, rr-ll-1), ",");
                    for (auto& arg : args)
                        push_back(arg+";", &inner, raw);
                }
                type = "func";
                end--;
                // j is the start index of s in src, l is the offset of s
                i = j + l;
                continue;
            }
            try_parse_define(s);
            if (raw) attrs["raw"] = "1";
            return;
        } else {
            push_back(s, nullptr, raw);
            i = k-1;
            continue;
        }
    }
}


void KernelIR::move_loop_back() {
    vector<uint> cid(children.size());
    uint num=0;
    for (uint i=0; i<children.size(); i++)
        if (children[i]->type != "loop" || children[i]->has_attr("raw"))
            num++;
    for (uint i=0,j=0,k=0; i<children.size(); i++) {
        if (children[i]->type != "loop" || children[i]->has_attr("raw"))
            cid[i] = j++;
        else
            cid[i] = num + k++;
    }
    vector<unique_ptr<KernelIR>> cb(children.size());
    for (uint i=0; i<children.size(); i++)
        cb[cid[i]] = move(children[i]);
    children = move(cb);
}


void KernelIR::replace(const vector<pair<string,string>>& replace_vars, bool equal, bool remove_define) {
    string& lvalue = get_attr("lvalue");
    string& code = get_attr("code");
    string& rvalue = get_attr("rvalue");

    int replace_time = 0;
    int max_time = 1;
    while (replace_time<max_time) {
    replace_time++;
    bool replaced = false;

    if (lvalue.size()) {
        if (type=="loop") {
            for (auto& p : replace_vars)
                if (p.first != p.second)
                    if (startswith(lvalue, p.first, 0, equal)) {
                        lvalue = p.second + lvalue.substr(p.first.size());
                        replaced = true;
                        break;
                    }
        } else {
            bool inside_loop = father!=nullptr && (father->type=="loop" || flist==&father->inner);
            for (auto& p : replace_vars)
                if (p.first != p.second)
                    if (startswith(lvalue, p.first, 0, equal)) {
                        // remove this define if matched
                        if (remove_define && !inside_loop) {
                            erase();
                            return;
                        } else {
                            del_scope();
                            lvalue = p.second + lvalue.substr(p.first.size());
                            add_scope();
                            replaced = true;
                        }
                        break;
                    }
        }
    }
    string* ss[2] = {&code, &rvalue};
    for (int p=0; p<2; p++) {
        auto& code = *ss[p];
        for (uint i=0; i<code.size(); i++) {
            while (i<code.size() && !isvar(code[i])) i++;
            if (i >= code.size()) break;
            uint j=i+1;
            while (j<code.size() && isvar(code[j])) j++;
            for (auto& p : replace_vars)
                if (p.first != p.second) {
                    if (j-i>=p.first.size() && startswith(code, p.first, i, equal, j)) {
                        code.erase(i, p.first.size());
                        code.insert(i, p.second);
                        j = j-p.first.size()+p.second.size();
                        replaced = true;
                        break;
                    }
                }
            i = j;
        }
    }

    if (!replaced) break;
    }

    ASSERT(max_time==1 || replace_time<max_time) << "replace encount infinit loop" << replace_vars << attrs;
    // replace children
    // reverse for prevent delete error
    for_each_rev([&](unique_ptr<KernelIR>& c) {
        c->replace(replace_vars, equal);
    });
}


void KernelIR::rename_loop_index() {
    vector<KernelIR*> irs(1, this);
    for (uint i=0; i<irs.size(); i++) {
        KernelIR* ir = irs[i];
        auto& rvalue = ir->get_attr("rvalue");
        auto& lvalue = ir->get_attr("lvalue");
        if (ir->type == "loop" && rvalue.size()) {
            if (startswith(rvalue, "range")) {
                auto& loop_id = ir->get_attr("loop_id");
                loop_id = rvalue.substr(5);
                ir->replace({{lvalue, "id"+rvalue.substr(5)}}, true);
            } else {
                // TODO
                LOGvvvv << "Unhandled loop var" << rvalue;
            }
        }
        for (auto& c : ir->children)
            if (c->type == "loop")
                irs.push_back(c.get());
    }
}


void KernelIR::merge_loop() {
    unordered_map<string, KernelIR*> loops;
    for (int i=(int)children.size()-1; i>=0; i--) {
        auto& loop = children[i];
        if (loop->type != "loop")
            continue;
        auto& loop_id = loop->get_attr("loop_id");
        if (!loop_id.size()) continue;
        auto iter = loops.find(loop_id);
        if (iter == loops.end()) {
            loops[loop_id] = loop.get();
            continue;
        }
        auto* mloop = iter->second;
        ASSERT(mloop->check_attr("loop_id", loop_id));
        mloop->insert(0, loop->children);
        children.erase(children.begin()+i);
    }
    for (auto& kv : loops)
        kv.second->merge_loop();
}

void KernelIR::solve_conflict_define() {
    unordered_set<string> defs;
    for (size_t i=0; i<children.size(); i++) {
        auto& c = children[i];
        if (c->type == "define") {
            auto lvalue = c->get_attr("lvalue");
            if (lvalue.size()==0)
                continue;
            if (defs.count(lvalue)) {
                // add _ to conflict definitions
                string new_def = lvalue + '_';
                while (defs.count(new_def))
                    new_def += '_';
                LOGvvv << "Conflict define" << c->to_string() << "change to" << new_def;
                for (size_t j=i; j<children.size(); j++)
                    children[j]->replace({{lvalue, new_def}}, true, false);
                defs.insert(new_def);
            } else
                defs.insert(lvalue);
        } else
        if (c->type == "loop")
            c->solve_conflict_define();
    }
}

void KernelIR::expand_empty_block() {
    for (uint i=0; i<children.size(); i++) {
        auto& loop = children[i];
        if (loop->type != "loop") continue;
        if (loop->has_attr("loop_id")) {
            loop->expand_empty_block();
            continue;
        }
        if (loop->has_attr("rvalue"))
            continue;
        insert(i+1, loop->children);
        // use children[i] instead of loop
        children[i]->erase();
        i--;
    }
}

void KernelIR::check_unused() {
    if (has_attr("raw")) return;
    attrs["used"] = "";
    const char* ss[] = {"code", "rvalue", "rvalue2"};
    for (const char* s : ss) {
        auto& code = get_attr(s);
        for (uint i=0; i<code.size(); i++) {
            while (i<code.size() && !isvar(code[i])) i++;
            if (i >= code.size()) break;
            uint j=i+1;
            while (j<code.size() && isvar(code[j])) j++;
            string var = code.substr(i,j-i);
            if (var=="void") {
                if (type=="") {
                    // remove (void)xxx;
                    code = "";
                    break;
                }
            }
            auto* def = find_define(var);
            if (def) {
                def->attrs["used"] = "1";
            }
            i = j;
        }
    }
    for_each([&](unique_ptr<KernelIR>& c) {
        c->check_unused();
    });
}

void KernelIR::find_used(KernelIR* def, vector<KernelIR*>& used) {
    if (has_attr("raw")) return;
    const char* ss[] = {"code", "rvalue", "rvalue2"};
    for (const char* s : ss) {
        auto& code = get_attr(s);
        for (uint i=0; i<code.size(); i++) {
            while (i<code.size() && !isvar(code[i])) i++;
            if (i >= code.size()) break;
            uint j=i+1;
            while (j<code.size() && isvar(code[j])) j++;
            string var = code.substr(i,j-i);
            auto* def2 = find_define(var);
            if (def2 == def)
                used.push_back(this);
            i = j;
        }
    }
    for_each([&](unique_ptr<KernelIR>& c) {
        c->find_used(def, used);
    });
}

vector<KernelIR*> KernelIR::find_used() {
    vector<KernelIR*> used;
    if (father) father->find_used(this, used);
    return used;
}


bool KernelIR::remove_unused() {
    bool has_unused = false;
    for_each_rev([&](unique_ptr<KernelIR>& c) {
        has_unused |= c->remove_unused();
    });
    if (type=="define" && check_attr("used", "")) {
        LOGvvvv << "Remove unused value:" << attrs["lvalue"];
        erase();
        return true;
    }
    return has_unused;
}

void KernelIR::remove_all_unused() {
    while (1) {
        check_unused();
        if (!remove_unused())
            break;
    }
}

void KernelIR::remove_intermediate(const unordered_set<string>& names) {
    const char* ss[] = {"code", "lvalue", "rvalue"};
    bool need_re_parse = false;
    for (const char* s : ss) {
        auto& code = get_attr(s);
        for (uint i=0; i<code.size(); i++) {
            while (i<code.size() && !isvar(code[i])) i++;
            if (i >= code.size()) break;
            uint j=i+1;
            while (j<code.size() && isvar(code[j])) j++;
            if (j<code.size() && code[j]=='[') {
                // find xxx[...]
                while (j<code.size() && code[j]!=']') j++;
                j++;
            }
            uint k=j-1;
            if (code[k] == ']') {
                // xxxp[...] -> xxxd
                while (k>=i && code[k]!='[') k--;
                k--;
                if (k>=i && code[k]=='p' && names.count(code.substr(i,k-i))) {
                    code[k] = 'd';
                    for (uint l=k+1; l<j; l++) code[l] = ' ';
                    if (i==0 && s==string("code")) {
                        // xxxd = xxx -> auto xxxd = xxx
                        code = "auto " + code;
                        need_re_parse = true;
                        j += 5;
                    }
                }
            } else
            if (code[k] == 'p' && string(s)=="lvalue" && type=="define") {
                if (names.count(code.substr(i,k-i))) {
                    erase();
                    return;
                }
            } else 
            if (code[k] == 'p' && string(s)=="code" && type=="") {
                if (names.count(code.substr(i,k-i))) {
                    // xxxp -> 0
                    for (uint l=i; l<j; l++)
                        code[l] = l==i ? '0' : ' ';
                }
            } 
            i=j-1;
        }
    }
    if (need_re_parse) {
        try_parse_define(string(attrs["code"]));
        return;
    }
    for_each_rev([&](unique_ptr<KernelIR>& c) {
        c->remove_intermediate(names);
    });
}


void KernelIR::split_loop(int i, int j) {
    if (type=="loop" && check_attr("loop_id", S(i))) {
        auto sj = S(j);
        auto si = S(i);
        auto& dtype = get_attr("dtype");
        auto& rvalue2 = get_attr("rvalue2");
        auto& lvalue = get_attr("lvalue");
        auto c = move_out_children();
        // set stride of loop i
        rvalue2 = "stride" + si;
        // change lvalue++ -> lvalue+=rvalue2
        ASSERT(inner.size()>=3);
        ASSERT(lvalue == "id"+si);
        inner[2]->attrs["code"] = lvalue+"+="+rvalue2+";";
        push_back("for ("+dtype+" id"+sj+"=0; id"+sj+"<range"+sj+"; id"+sj+"++) {}");
        auto& sloop = children.back();
        int range=0, stride=0;
        if (get_number("range"+si, range) && get_number("stride"+si, stride) && (range%stride==0))
            push_back(dtype+" range"+sj+" = "+S(stride)+";", &inner);
        else {
            ASSERT(range != -1 && stride != -1) << range << stride << si;
            push_back(dtype+" range"+sj+" = ::min(range"+si+"-id"+si+", stride"+si+");", &inner);
        }
        sloop->attrs["loop_id"] = sj;
        sloop->attrs["split_id"] = si;
        sloop->insert(0, c);
        sloop->replace({{"id"+si, "(id"+si+"+id"+sj+")"}}, true);
        return;
    }
    for_each([&](unique_ptr<KernelIR>& c) {
        c->split_loop(i,j);
    });
}

void KernelIR::resplit() {
    if (has_attr("resplited")) return;
    ASSERT(type=="loop");
    attrs["resplited"] = "1";
    auto& rvalue2 = get_attr("rvalue2");
    auto& lvalue = get_attr("lvalue");
    auto& rvalue = get_attr("rvalue");
    auto& dtype = get_attr("dtype");
    ASSERT(rvalue2.size());
    ASSERT(inner.size()>3 && startswith(inner[3]->get_attr("lvalue"), "range"));
    ASSERT(startswith(inner[3]->get_attr("rvalue"), "::min")) <<
        "No need to resplit";
    
    // delete prev inner code(init and condition)
    inner[0]->erase();
    inner[0]->erase();
    // condition
    push_front(lvalue+"+"+rvalue2+"<="+rvalue+";", &inner);
    // init
    push_front(lvalue+"=0;", &inner);
    // define
    push_front(dtype+" "+lvalue+" = 0;", &before);
    
    int num=0;
    if (get_number(rvalue2, num)) {
        // range = number;
        inner[3]->attrs["rvalue"] = S(num);
    } else {
        ASSERT(num == -2);
        // range = rvalue2;
        inner[3]->attrs["rvalue"] = rvalue2;
    }
    // add if and clone children
    push_back("if ("+lvalue+"<"+rvalue+") {}", &after);
    after.back()->push_back(dtype+" "+inner[3]->get_attr("lvalue")+" = "+rvalue+"-"+lvalue+";");
    for (auto &c : children)
        after.back()->push_back(c->clone());
}

} // jittor
