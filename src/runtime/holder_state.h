#pragma once
#include <iterator>
#include <list>
#include "common.h"

namespace jittor {

struct VarHolder;

// One owner for the Python-held roots and the weak-sync traversal boundary.
// Mutations retain the existing serialized runtime/GIL requirement.
class RuntimeHolderState {
public:
    using Holders = std::list<VarHolder*>;
    using Iterator = Holders::iterator;

    RuntimeHolderState() : sync_cursor_(holders_.end()) {}
    RuntimeHolderState(const RuntimeHolderState&) = delete;
    RuntimeHolderState& operator=(const RuntimeHolderState&) = delete;

    const Holders& holders() const { return holders_; }
    bool contains(Iterator it) const { return it != holders_.end(); }

    Iterator add(VarHolder* holder) {
        holders_.push_front(holder);
        return holders_.begin();
    }

    void erase(Iterator& it) {
        if (it == holders_.end()) return;
        if (it == sync_cursor_) ++sync_cursor_;
        holders_.erase(it);
        it = holders_.end();
    }

    VarHolder* peek_pending() const {
        if (sync_cursor_ == holders_.begin()) return nullptr;
        return *std::prev(sync_cursor_);
    }

    // The executor checks the candidate's target cutoff before consuming it.
    void consume_pending() {
        if (sync_cursor_ != holders_.begin()) --sync_cursor_;
    }

private:
    Holders holders_;
    Iterator sync_cursor_;
};

EXTERN_LIB RuntimeHolderState& runtime_holder_state();

} // namespace jittor
