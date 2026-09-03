// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cstring>
#include "misc/nano_string.h"

namespace jittor {

// The index field and the tables it indexes must agree on how many entries
// exist. They did not: the tables were built for 256 and the field held 7 bits,
// so entry 128 would have been written as entry 0 -- not a crash, an alias.
// `set()` masks, which is what makes the failure silent.
JIT_TEST(nano_string_index_addresses_every_table_slot) {
    NanoString ns;
    // neighbours, so that a too-narrow field is not mistaken for a field that
    // simply overflows into the next one
    ns.set(NanoString::_dsize, 3, NanoString::_dsize_nbits);
    ns.set(NanoString::_float, 1);

    ns.set(NanoString::_index, ns_max_size-1, NanoString::_index_nbits);
    ASSERTop(ns.index(),==,(NanoString::ns_t)(ns_max_size-1))
        << "the index field cannot address the last table entry, so a"
           " registration there silently aliases an existing one";
    ASSERTop(ns.dsize(),==,8);
    ASSERT(ns.is_float());

    // and the whole range round-trips, not just its ends
    for (NanoString::ns_t i=0; i<(NanoString::ns_t)ns_max_size; i++) {
        ns.set(NanoString::_index, i, NanoString::_index_nbits);
        ASSERTop(ns.index(),==,i);
    }
}

// The two registration limits, asked the way init_ns() asks them.
JIT_TEST(nano_string_registration_refuses_what_it_cannot_hold) {
    // one past the last slot
    expect_error([&]() { ns_check_registration(ns_max_size, "spill"); });
    // ns_max_len characters leaves no room for the terminator, so it would
    // write the first byte of the *next* entry's name
    string too_long(ns_max_len, 'x');
    expect_error([&]() { ns_check_registration(0, too_long.c_str()); });

    // the largest name that does fit must still be accepted, or the check is
    // just a smaller limit wearing an assertion
    string longest(ns_max_len-1, 'x');
    ns_check_registration(ns_max_size-1, longest.c_str());
}

// Every registered name must read back as itself. This is the assertion that
// notices a name slot overwritten by its neighbour -- the failure mode the
// length check above exists to prevent, seen from the other side.
JIT_TEST(nano_string_names_round_trip) {
    #define CHECK_NS_NAME(T) \
        ASSERTop(string(ns_##T.to_cstring()),==,string(#T)); \
        ASSERTop(ns_##T.len(),==,(int)strlen(#T)); \
        ASSERT(__string_to_ns.at(#T) == ns_##T);
    FOR_ALL_NS(CHECK_NS_NAME);
    #undef CHECK_NS_NAME

    // distinct entries, i.e. nothing aliased onto anything else
    unordered_set<NanoString::ns_t> seen;
    #define CHECK_NS_INDEX(T) \
        ASSERT(seen.insert(ns_##T.index()).second) \
            << "duplicate NanoString index for" << #T;
    FOR_ALL_NS(CHECK_NS_INDEX);
    #undef CHECK_NS_INDEX
}

}
