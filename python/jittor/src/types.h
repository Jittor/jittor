// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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
#include <new>
#include <type_traits>

namespace jittor {

typedef int8_t int8;
typedef int16_t int16;
typedef int int32;
typedef long long int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef float float32;
typedef double float64;
typedef uint32_t uint;
typedef uint32 OpId;

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

// A vector with room for the common case inside its owner. Node edge tables
// usually contain one producer or a handful of consumers, so allocating a
// separate list node for every edge is disproportionately expensive.
template <class T, size_t InlineCapacity>
class SmallVector {
    static_assert(InlineCapacity > 0, "SmallVector needs inline storage");
    typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type Storage;

    size_t size_ = 0;
    size_t capacity_ = InlineCapacity;
    Storage inline_storage_[InlineCapacity];
    T* data_ = inline_data();

    T* inline_data() { return reinterpret_cast<T*>(inline_storage_); }
    const T* inline_data() const { return reinterpret_cast<const T*>(inline_storage_); }

    void destroy_elements() {
        for (size_t i = 0; i < size_; ++i) data_[i].~T();
        size_ = 0;
    }

    void release_storage() {
        if (data_ != inline_data()) ::operator delete(data_);
        data_ = inline_data();
        capacity_ = InlineCapacity;
    }

    void grow(size_t minimum) {
        size_t next_capacity = capacity_ * 2;
        if (next_capacity < minimum) next_capacity = minimum;
        T* next = static_cast<T*>(::operator new(sizeof(T) * next_capacity));
        size_t moved = 0;
        try {
            for (; moved < size_; ++moved)
                new (next + moved) T(std::move_if_noexcept(data_[moved]));
        } catch (...) {
            while (moved) next[--moved].~T();
            ::operator delete(next);
            throw;
        }
        size_t old_size = size_;
        destroy_elements();
        if (data_ != inline_data()) ::operator delete(data_);
        data_ = next;
        size_ = old_size;
        capacity_ = next_capacity;
    }

    void move_from(SmallVector&& other) {
        if (!other.using_inline_storage()) {
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = other.inline_data();
            other.size_ = 0;
            other.capacity_ = InlineCapacity;
            return;
        }
        reserve(other.size_);
        for (T& value : other) emplace_back(std::move(value));
        other.clear();
    }

public:
    typedef T* iterator;
    typedef const T* const_iterator;

    SmallVector() = default;

    SmallVector(const SmallVector& other) {
        reserve(other.size_);
        for (const T& value : other) emplace_back(value);
    }

    SmallVector(SmallVector&& other) noexcept(std::is_nothrow_move_constructible<T>::value) {
        move_from(std::move(other));
    }

    ~SmallVector() {
        destroy_elements();
        release_storage();
    }

    SmallVector& operator=(const SmallVector& other) {
        if (this == &other) return *this;
        clear();
        reserve(other.size_);
        for (const T& value : other) emplace_back(value);
        return *this;
    }

    SmallVector& operator=(SmallVector&& other) noexcept(std::is_nothrow_move_constructible<T>::value) {
        if (this == &other) return *this;
        destroy_elements();
        release_storage();
        move_from(std::move(other));
        return *this;
    }

    iterator begin() { return data_; }
    const_iterator begin() const { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator end() const { return data_ + size_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }
    bool using_inline_storage() const { return data_ == inline_data(); }

    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    T& front() { return data_[0]; }
    const T& front() const { return data_[0]; }
    T& back() { return data_[size_ - 1]; }
    const T& back() const { return data_[size_ - 1]; }

    void reserve(size_t requested) {
        if (requested > capacity_) grow(requested);
    }

    template <class... Args>
    void emplace_back(Args&&... args) {
        if (size_ == capacity_) grow(size_ + 1);
        new (data_ + size_) T(std::forward<Args>(args)...);
        ++size_;
    }

    void push_back(const T& value) { emplace_back(value); }
    void push_back(T&& value) { emplace_back(std::move(value)); }

    void pop_back() { data_[--size_].~T(); }
    void clear() { destroy_elements(); }

    void resize(size_t requested) {
        if (requested < size_) {
            while (size_ > requested) pop_back();
            return;
        }
        reserve(requested);
        while (size_ < requested) emplace_back();
    }

    iterator erase(iterator position) {
        size_t index = static_cast<size_t>(position - data_);
        for (size_t i = index; i + 1 < size_; ++i)
            data_[i] = std::move(data_[i + 1]);
        pop_back();
        return data_ + index;
    }
};

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
    SmallVector<To, 2> *ptr;
    Caster(SmallVector<To, 2>* ptr) : ptr(ptr) {};
    struct Iter {
        typename SmallVector<To, 2>::iterator iter;
        Iter(typename SmallVector<To, 2>::iterator iter) : iter(iter) {}
        T operator*() { return iter->operator T(); }
        Iter& operator++() { ++iter; return *this; }
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

#define JPU(x) ;

} // jittor
