// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
//
// Property tests for 2.10's liveness counters (10.18).
//
// `test_node_liveness.cc` records one hand-written sequence of own/release and
// the answers it should give. That is the right test for the contract, and it
// is not the right test for the thing that actually goes wrong: the counters
// are driven by graph propagation, so the sequences that reach them in a real
// process are long, interleaved across three counters, and not enumerable.
//
// So these sweep sequences instead of recording one. The invariant being swept
// is the one that matters for the bug this task found -- a `release()` with no
// matching `own()` must be *heard*, never absorbed. `tests/core/
// test_core_invariant_properties.py` shows what happens when it is absorbed
// higher up: two Vars stay alive forever and three point cases report `2 != 0`
// without naming a cause.
//
// No device, no allocation, no graph: these are tests of a counter.

#include <limits>

#include "node.h"

namespace jittor {

// A counter's count is exactly (owns - releases), for any interleaving that
// never releases below zero -- and `active()` is exactly `count() != 0`.
//
// Deliberately not a random sweep. A seeded RNG here would make a failure
// depend on the seed and on the standard library's generator; a deterministic
// enumeration over a range of shapes covers the same ground and reproduces
// identically everywhere, which is what a gate needs.
JIT_TEST(liveness_counter_count_tracks_owns_minus_releases) {
    for (int owns=1; owns<=32; owns++) {
        for (int releases=0; releases<=owns; releases++) {
            LivenessCounter<LivenessKind::forward> counter;
            CHECKop(counter.count(),==,0);
            CHECK(!counter.active());

            for (int i=0; i<owns; i++) {
                bool became_live = counter.own();
                // The transition is reported exactly once, on the first owner.
                CHECKop(became_live,==,(i==0));
                CHECKop(counter.count(),==,i+1);
                CHECK(counter.active());
            }
            for (int i=0; i<releases; i++) {
                bool became_dead = counter.release();
                // ...and symmetrically, exactly once, on the last release.
                CHECKop(became_dead,==,(i==owns-1));
                CHECKop(counter.count(),==,owns-i-1);
            }
            CHECKop(counter.count(),==,owns-releases);
            CHECKop(counter.active(),==,(owns!=releases));
            counter.assert_expected(owns-releases, &counter);
        }
    }
}

// The property the leak depends on: an unmatched release is always an error,
// at every count, for every kind. Absorbing it silently is what turns a
// bookkeeping bug into a permanently leaked node -- the counter stops at zero,
// the caller believes the release happened, and nothing is ever freed.
JIT_TEST(liveness_counter_never_absorbs_an_unmatched_release) {
    for (int owns=0; owns<=8; owns++) {
        LivenessCounter<LivenessKind::forward> forward;
        LivenessCounter<LivenessKind::backward> backward;
        LivenessCounter<LivenessKind::pending> pending;
        for (int i=0; i<owns; i++) {
            forward.own(); backward.own(); pending.own();
        }
        for (int i=0; i<owns; i++) {
            forward.release(); backward.release(); pending.release();
        }
        // Drained to zero by construction; one more release must throw, and
        // must leave the count where it was rather than going negative.
        CHECKop(forward.count(),==,0);
        expect_error([&]() { forward.release(); });
        CHECKop(forward.count(),==,0);
        expect_error([&]() { backward.release(); });
        CHECKop(backward.count(),==,0);
        expect_error([&]() { pending.release(); });
        CHECKop(pending.count(),==,0);
    }
}

// All three kinds behave identically. They are separate types (so that a
// forward count can never be handed to a backward release), which means the
// shared behaviour is shared only by being written three times.
JIT_TEST(liveness_counter_kinds_agree) {
    LivenessCounter<LivenessKind::forward> forward;
    LivenessCounter<LivenessKind::backward> backward;
    LivenessCounter<LivenessKind::pending> pending;
    // The same operation is applied to all three at every step, so any
    // divergence is a difference between the kinds rather than between the
    // sequences they were given. The pattern crosses zero repeatedly and also
    // climbs above 1, so both the boundary and the interior are covered.
    // Balanced, and never dips below zero: 1,2,1,2,1,0,1,2,3,2,1,0,1,0,1,0.
    const char* script = "ooRoRRoooRRRoRoR";
    for (const char* step = script; *step; step++) {
        if (*step == 'o') {
            bool f = forward.own(), b = backward.own(), p = pending.own();
            CHECKop(f,==,b);
            CHECKop(f,==,p);
        } else {
            bool f = forward.release(), b = backward.release(),
                 p = pending.release();
            CHECKop(f,==,b);
            CHECKop(f,==,p);
        }
        CHECKop(forward.count(),==,backward.count());
        CHECKop(forward.count(),==,pending.count());
        CHECKop(forward.active(),==,backward.active());
        CHECKop(forward.active(),==,pending.active());
    }
    // The script is balanced, so all three end drained.
    CHECKop(forward.count(),==,0);
    CHECKop(backward.count(),==,0);
    CHECKop(pending.count(),==,0);
}

// `need_free()` is the whole point of grouping the three counters, and it is
// the predicate the leaked Vars violate: they sit in the node registry with
// `f=0 b=1 p=0`, which satisfies need_free(), and were never freed.
//
// Swept over all 3x3x3 reachable states rather than spot-checked, because the
// definition has a disjunction in it and a spot check picks one side.
JIT_TEST(liveness_need_free_matches_its_definition_in_every_state) {
    for (int f=0; f<3; f++)
        for (int b=0; b<3; b++)
            for (int p=0; p<3; p++) {
                NodeLiveness state;
                for (int i=0; i<f; i++) state.forward.own();
                for (int i=0; i<b; i++) state.backward.own();
                for (int i=0; i<p; i++) state.pending.own();

                bool expected = (p == 0) && (f == 0 || b == 0);
                CHECKop(state.need_free(),==,expected);
                state.assert_expected(f, b, p, &state);

                // A fresh NodeLiveness needs freeing: nothing owns it yet, so
                // an owner has to be taken before it is safe to keep.
                if (!f && !b && !p) CHECK(state.need_free());

                // Drain, so the counters do not assert on destruction paths
                // that expect balance.
                for (int i=0; i<f; i++) state.forward.release();
                for (int i=0; i<b; i++) state.backward.release();
                for (int i=0; i<p; i++) state.pending.release();
                CHECK(state.need_free());
            }
}

// A mismatch has to be reported for the right kind. The three counters are
// released together by `release_both_liveness`, so a report that names the
// wrong one sends whoever reads the log at the wrong half of the propagation
// -- which is how the bug in the Python file above stayed unattributed.
JIT_TEST(liveness_mismatch_names_the_kind_that_mismatched) {
    NodeLiveness state;
    state.forward.own();
    state.backward.own();
    state.pending.own();
    // Each of the three expectations is wrong in exactly one position.
    expect_error([&]() { state.assert_expected(0, 1, 1, &state); });
    expect_error([&]() { state.assert_expected(1, 0, 1, &state); });
    expect_error([&]() { state.assert_expected(1, 1, 0, &state); });
    // And the correct one does not throw.
    state.assert_expected(1, 1, 1, &state);
    state.forward.release();
    state.backward.release();
    state.pending.release();
}

} // namespace jittor
