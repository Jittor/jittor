// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cctype>
#include "utils/str_utils.h"

namespace jittor {


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
    int pos = 0, pos_next;
    while (1) {
        pos_next = s.find(sep, pos);
        if (pos_next == (int)string::npos || (int)ret.size() == max_split-1) {
            ret.push_back(s.substr(pos));
            return ret;
        }
        ret.push_back(s.substr(pos, pos_next-pos));
        pos = pos_next + sep.size();
    }
    ASSERT(max_split==0);
    return ret;
}

string strip(const string& s) {
    int i=0;
    while (i<s.size() && (s[i]==' ' || s[i]=='\t' || s[i]=='\n' || s[i]=='\r')) i++;
    int j = s.size();
    while (j>i && (s[j-1]==' ' || s[j-1]=='\t' || s[j-1]=='\n' || s[j-1]=='\r')) j--;
    return s.substr(i,j-i);
}

string format(const string& s, const vector<string>& v) {
    string ss;
    for (int i=0; i<s.size(); i++) {
        if (s[i] == '$') {
            int j = s[i+1] - '0';
            ss += v.at(j);
            i ++;
            continue;
        } else
            ss += s[i];
    }
    return ss;
}

string join(const vector<string>& vs, const string& x) {
    string s;
    for (int i=0; i<vs.size(); i++) {
        s += vs[i];
        if (i!=vs.size()-1)
            s += x;
    }
    return s;
}

string replace(const string& a, const string& b, const string& c) {
    auto vs = split(a, b);
    return join(vs, c);
}

static inline bool isvar(char x) { return isalnum(x) || x == '_' || x == ':'; }

vector<string> token_split(const string& s, bool exclude_comments) {
    vector<string> ss;
    if (!s.size()) return ss;
    ss.push_back("");
    for (int i = 0; i < s.size(); i++) {
        if (exclude_comments) {
            if (s[i] == '/' && s[i+1] == '/') {
                i = s.find('\n', i);
                if (i == string::npos)
                    return ss;
            }
            if (s[i] == '/' && s[i+1] == '*') {
                i = s.find("*/", i);
                if (i == string::npos)
                    return ss;
                i += 1;
                continue;
            }
        }
        if (i && (isvar(s[i]) != isvar(s[i-1])))
            ss.push_back("");
        ss.back() += s[i];
    }
    return ss;
}

static void parse_reg(const string& src, 
    vector<string>& patterns,
    vector<int>& arg_id,
    bool match_whitespace=true) {
    patterns.clear();
    arg_id.clear();
    patterns.push_back("");
    for (int j=0; j<src.size(); j++) {
        if (src[j] == '$') {
            j++;
            arg_id.push_back(src[j]-'0');
            patterns.push_back("");
            continue;
        }
        if (match_whitespace || !isspace(src[j]))
            patterns.back() += src[j];
    }
}

int token_replace(vector<string>& tokens, int i, const string& src, const string& dst, bool match_whitespace) {
    ASSERT(src.at(0) != '$' && src.at(src.size()-1) != '$' && 
        src.at(src.size()-2) != '$') << "illegal src:" << src;
    vector<string> patterns;
    vector<int> arg_id;
    vector<string> patterns2;
    vector<int> arg_id2;
    unordered_map<int, string> args;
    parse_reg(src, patterns, arg_id, match_whitespace);
    parse_reg(dst, patterns2, arg_id2);

    int start_i, start_pos, end_i, end_pos;
    int c_i = i, c_pos = 0;
    int match_i, match_pos;
    string c_arg;

    auto next = [&tokens](int &c_i, int &c_pos) {
        c_pos ++;
        if (c_pos >= tokens[c_i].size()) {
            c_pos = 0;
            c_i ++;
            if (c_i >= tokens.size())
                return false;
        }
        return true;
    };

    auto match = [&](int c_i, int c_pos, const string& pat) -> bool {
        for (int i=0; i<pat.size(); i++) {
            while (!match_whitespace && isspace(tokens[c_i][c_pos])) 
                next(c_i, c_pos);
            if (tokens[c_i][c_pos] != pat[i])
                return false;
            next(c_i, c_pos);            
        }
        match_i = c_i;
        match_pos = c_pos;
        return true;
    };

    for (int j=0; j<patterns.size(); j++) {
        int ok = 0;
        while (c_i < tokens.size()) {
            while (c_pos < tokens[c_i].size()) {
                if (match(c_i, c_pos, patterns[j])) {
                    ok = 1;
                    break;
                }
                c_arg += tokens[c_i][c_pos];
                c_pos ++;
            }
            if (ok) break;
            c_i ++;
            c_pos = 0;
        }
        CHECK(ok) << "Pattern not match:" << patterns[j] << j;
        if (j == 0) {
            start_i = c_i;
            start_pos = c_pos;
        }
        if (j) {
            args[arg_id[j-1]] = c_arg;
        }
        c_arg = "";
        c_i = match_i;
        c_pos = match_pos;
        if (j == patterns.size()-1) {
            end_i = c_i;
            end_pos = c_pos;
        }
    }
    string new_src;
    for (int j=0; j<patterns2.size(); j++) {
        if (j) new_src += args[arg_id2.at(j-1)];
        new_src += patterns2[j];
    }
    if (start_i == end_i) {
        tokens[start_i] = tokens[start_i].substr(0, start_pos) +
            new_src + tokens[start_i].substr(end_pos);
    } else {
        tokens[start_i] = tokens[start_i].substr(0, start_pos)
            + new_src;
        tokens[end_i] = tokens[end_i].substr(end_pos);
        for (int j=start_i+1; j<end_i; j++)
            tokens[j] = "";
    }
    return end_i;
}

string token_replace(const string& s, const string& src, const string& dst, bool match_whitespace) {
    vector<string> ss{s};
    token_replace(ss, 0, src, dst, match_whitespace);
    return join(ss, "");
}

string token_replace_all(const string& s, const string& src, const string& dst) {
    auto ss = token_split(s);
    int pos = 0;
    while (pos < ss.size()) {
        try {
            pos = token_replace(ss, pos, src, dst) + 1;
        } 
        catch(const std::exception& e) {
            return join(ss, "");
        }
    }
    return join(ss, "");
}

} // jittor