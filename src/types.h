// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <vector>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace jittor {

typedef int8_t int8;
typedef int16_t int16;
typedef int int32;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef float float32;
typedef double float64;
typedef uint32_t uint;

using string = std::string;
using std::move;
template <class T> using vector = std::vector<T>;
template <class T> using list = std::list<T>;
template <class T> using set = std::set<T>;
template <class T> using shared_ptr = std::shared_ptr<T>;
template <class T> using unique_ptr = std::unique_ptr<T>;
template <class T> using unordered_set = std::unordered_set<T>;
template <class Ta, class Tb> using pair = std::pair<Ta,Tb>;
template <class Ta, class Tb> using map = std::map<Ta,Tb>;
template <class Ta, class Tb> using unordered_map = std::unordered_map<Ta,Tb>;

struct Node;
struct Var;
struct Op;
struct Allocator;
struct Executor;
struct VarHolder;
struct VarPtr;
struct FusedOp;
struct OpCompiler;
struct PassManager;
struct Pass;
struct TunerManager;
struct Tuner;
struct NanoString;

typedef map<string,string> map_string;
typedef map<string,int> loop_options_t;
typedef map<string,vector<int>> loop_option_candidates_t;
typedef void (*jit_op_entry_t)(Op* op);

template<class T>
T clone(const T& a) { return a; }

#define function_alias(A, B) \
template <typename... Args> \
auto B(Args&&... args) -> decltype(A(std::forward<Args>(args)...)) { \
  return A(std::forward<Args>(args)...); \
}

function_alias(std::to_string, S);

template <class Ta, class Tb>
std::ostream& operator<<(std::ostream& os, const pair<Ta, Tb>& p) {
    return os << '(' << p.first << ',' << p.second << ')';
}

// print tuple function
namespace aux{
template<std::size_t...> struct seq{};

template<std::size_t N, std::size_t... Is>
struct gen_seq : gen_seq<N-1, N-1, Is...>{};

template<std::size_t... Is>
struct gen_seq<0, Is...> : seq<Is...>{};

template<class Ch, class Tr, class Tuple, std::size_t... Is>
void print_tuple(std::basic_ostream<Ch,Tr>& os, Tuple const& t, seq<Is...>){
  using swallow = int[];
  (void)swallow{0, (void(os << (Is == 0? "" : ",") << std::get<Is>(t)), 0)...};
}
} // aux::

template<class Ch, class Tr, class... Args>
auto operator<<(std::basic_ostream<Ch, Tr>& os, std::tuple<Args...> const& t)
    -> std::basic_ostream<Ch, Tr>&
{
  os << "[";
  aux::print_tuple(os, t, aux::gen_seq<sizeof...(Args)>());
  return os << "]";
}


template <class T>
std::ostream& operator<<(std::ostream& os, unique_ptr<T>& ptr) {
    return os << *ptr;
}

template <class T>
std::ostream& operator<<(std::ostream& os, shared_ptr<T>& ptr) {
    return os << *ptr;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const unique_ptr<T>& ptr) {
    return os << *ptr;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const shared_ptr<T>& ptr) {
    return os << *ptr;
}

template <class T>
std::ostream& operator<<(std::ostream& os, vector<T>& input) {
    os << '[';
    for (auto& i: input) os << i << ",";
    return os << ']';
}

template <class T>
std::ostream& operator<<(std::ostream& os, list<T>& input) {
    os << '[';
    for (auto& i: input) os << i << ",";
    return os << ']';
}

template <class Ta, class Tb>
std::ostream& operator<<(std::ostream& os, map<Ta, Tb>& input) {
    os << '{';
    for (auto& i: input) os << i.first << ':' << i.second << ", ";
    return os << '}';
}

template <class T>
std::ostream& operator<<(std::ostream& os, const vector<T>& input) {
    os << '[';
    for (auto const& i: input) os << i << ",";
    return os << ']';
}

template <class T>
std::ostream& operator<<(std::ostream& os, const list<T>& input) {
    os << '[';
    for (auto const& i: input) os << i << ",";
    return os << ']';
}

template <class T>
std::ostream& operator<<(std::ostream& os, const set<T>& input) {
    os << '[';
    for (auto const& i: input) os << i << ",";
    return os << ']';
}

template <class T>
std::istream& operator>>(std::istream& is, vector<T>& out) {
    T value;
    while (is >> value)
        out.push_back(value);
    return is;
}

template <class Ta, class Tb>
std::ostream& operator<<(std::ostream& os, const map<Ta, Tb>& input) {
    os << '{';
    for (auto const& i: input) os << i.first << ':' << i.second << ", ";
    return os << '}';
}

template <class Ta, class Tb>
std::istream& operator>>(std::istream& is, map<Ta, Tb>& out) {
    Ta key;
    Tb value;
    while (is >> key >> value)
        out[key] = value;
    return is;
}

template <class Ta, class Tb>
std::istream& operator>>(std::istream& is, unordered_map<Ta, Tb>& out) {
    Ta key;
    Tb value;
    while (is >> key >> value)
        out[key] = value;
    return is;
}


template <class Ta, class Tb>
std::ostream& operator<<(std::ostream& os, const unordered_map<Ta, Tb>& input) {
    os << '{';
    for (auto const& i: input) os << i.first << ':' << i.second << ", ";
    return os << '}';
}

template <class T>
std::ostream& operator<<(std::ostream& os, const unordered_set<T>& input) {
    os << '{';
    for (auto const& i: input) os << i << ", ";
    return os << '}';
}

template <typename T, typename To>
struct Caster {
    list<To> *ptr;
    Caster(list<To>* ptr) : ptr(ptr) {};
    struct Iter {
        typename list<To>::iterator iter, next;
        Iter(typename list<To>::iterator iter)
            : iter(iter), next(std::next(iter)) {}
        T operator*() { return iter->operator T(); }
        Iter& operator++() { iter = next++; return *this; }
        Iter operator++(int) { auto tmp = *this; ++(*this); return tmp; }
        bool operator!=(Iter& other) { return iter != other.iter; }
    };
    Iter begin() const { return Iter(ptr->begin()); }
    Iter end() const { return Iter(ptr->end()); }
    size_t size() { return ptr->size(); }
    T front() { return ptr->front().operator T(); }
    T back() { return ptr->back().operator T(); }
};

template <typename T, typename To>
std::ostream& operator<<(std::ostream& os, const Caster<T,To>& input) {
    os << '[';
    for (const T i: input) os << i << ",";
    return os << ']';
}

} // jittor
