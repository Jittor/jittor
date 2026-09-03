// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <typeinfo>
#include <typeindex>
#include <utility>
#include "common.h"

namespace jittor {

struct OpInfo {
    string name, source_path, extra_flags;
    vector<pair<const std::type_info*, void*>> constructors;
    // string: var member name, uint64: var member offset
    vector<pair<string, uint64>> var_members;

    template<class To, class ...Ts> auto get_constructor() {
        typedef To (*func_t)(Ts...);
        const auto& tid = typeid(func_t);
        for (uint i=0; i<constructors.size(); i++)
            if (std::type_index(*(constructors[i].first)) == std::type_index(tid))
                return func_t(constructors[i].second);
        LOGf << "constructor" << name << tid.name() << "not found.";
        return func_t(nullptr);
    }
};

void op_registe(const OpInfo& op_info);
bool has_op(const string& name);
OpInfo get_op_info(const string& name);

/** An op constructor resolved on first call instead of at load time.
 *
 * The spelling this replaces was, at *namespace* scope, in 113 places:
 *
 *     static auto make_binary = get_op_info("binary")
 *         .get_constructor<VarPtr, Var*, Var*, NanoString>();
 *
 * `get_op_info` asserts the op is registered. Registration is itself a static
 * initialiser -- `gen_ops.cc`'s `int caller = (initer(), 0)` and
 * `op_utils.cc`'s `init()` -- in a *different* translation unit, and the
 * relative order of static initialisers across translation units is
 * unspecified by the standard. So "registered by the time this line runs" was
 * never a guarantee; it was a link order that happened to hold.
 *
 * When it does not hold, the failure is the worst available shape: the ASSERT
 * throws out of a static initialiser, before main, where no catch exists, so
 * the process calls std::terminate with no message naming the op, the file, or
 * the ordering. Deferring the lookup to the first *call* removes the question:
 * by then main is running and every registration has happened.
 *
 * (Function-local `static auto x = get_op_info(...)` inside an op constructor
 * was always fine -- it is already lazy -- and is left alone.)
 */
template<class To, class ...Ts>
struct OpConstructor {
    typedef To (*func_t)(Ts...);
    const char* name;
    //: Resolved once, then reused. Two threads racing here compute the same
    //: pointer from a map that is never written after static initialisation,
    //: so the duplicated work is the whole cost of the race.
    mutable func_t cached = nullptr;

    explicit OpConstructor(const char* name) : name(name) {}

    inline To operator()(Ts... args) const {
        if (!cached)
            cached = get_op_info(name).template get_constructor<To, Ts...>();
        return cached(std::forward<Ts>(args)...);
    }

    //: "is this op available at all" -- for the optional backend ops.
    inline explicit operator bool() const { return has_op(name); }
};

template<class To, class ...Ts>
inline OpConstructor<To, Ts...> op_constructor(const char* name) {
    return OpConstructor<To, Ts...>(name);
}

struct OpCompiler;
struct OpByType {
    unordered_set<string> types;
    virtual string expand_op(const vector<string>& args) = 0;
    virtual void post_pass(OpCompiler*) = 0;
};

extern vector<OpByType*> op_types;
int registe_op_type(OpByType*);

} // jittor