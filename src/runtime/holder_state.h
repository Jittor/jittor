#pragma once
#include <cstddef>
#include <iterator>
#include "core/common.h"

namespace jittor {

struct VarHolder;

struct HolderLink {
    HolderLink* prev = nullptr;
    HolderLink* next = nullptr;
    VarHolder* owner = nullptr;

    inline bool linked() const { return prev != nullptr; }
};

// Intrusive circular doubly-linked list of live VarHolder* roots.
// Preserves LIFO push_front order and std::list-compatible const iteration,
// without per-holder heap node allocations.
class HolderList {
public:
    class const_iterator {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = VarHolder*;
        using difference_type = std::ptrdiff_t;
        using pointer = VarHolder* const*;
        using reference = VarHolder* const&;

        const_iterator() : node_(nullptr) {}
        explicit const_iterator(const HolderLink* node) : node_(node) {}

        reference operator*() const { return node_->owner; }
        pointer operator->() const { return &node_->owner; }
        const_iterator& operator++() { node_ = node_->next; return *this; }
        const_iterator operator++(int) { auto tmp = *this; node_ = node_->next; return tmp; }
        const_iterator& operator--() { node_ = node_->prev; return *this; }
        const_iterator operator--(int) { auto tmp = *this; node_ = node_->prev; return tmp; }
        bool operator==(const const_iterator& other) const { return node_ == other.node_; }
        bool operator!=(const const_iterator& other) const { return node_ != other.node_; }

    private:
        const HolderLink* node_;
    };

    using iterator = const_iterator;

    HolderList() {
        head_.prev = &head_;
        head_.next = &head_;
        head_.owner = nullptr;
    }
    HolderList(const HolderList&) = delete;
    HolderList& operator=(const HolderList&) = delete;

    const_iterator begin() const { return const_iterator(head_.next); }
    const_iterator end() const { return const_iterator(&head_); }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

private:
    friend class RuntimeHolderState;

    HolderLink* sentinel() { return &head_; }
    const HolderLink* sentinel() const { return &head_; }
    const HolderLink* front_link() const { return head_.next; }

    void push_front(HolderLink& link, VarHolder* owner) {
        if (link.linked()) return;
        link.owner = owner;
        link.prev = &head_;
        link.next = head_.next;
        head_.next->prev = &link;
        head_.next = &link;
        ++size_;
    }

    void unlink(HolderLink& link) {
        if (!link.linked()) return;
        link.prev->next = link.next;
        link.next->prev = link.prev;
        link.prev = nullptr;
        link.next = nullptr;
        link.owner = nullptr;
        --size_;
    }

    void replace(HolderLink& old_link, HolderLink& new_link, VarHolder* new_owner) {
        if (!old_link.linked()) {
            new_link.prev = nullptr;
            new_link.next = nullptr;
            new_link.owner = nullptr;
            return;
        }
        new_link.owner = new_owner;
        new_link.prev = old_link.prev;
        new_link.next = old_link.next;
        new_link.prev->next = &new_link;
        new_link.next->prev = &new_link;
        old_link.prev = nullptr;
        old_link.next = nullptr;
        old_link.owner = nullptr;
    }

    HolderLink head_;
    size_t size_ = 0;
};

// One owner for the Python-held roots and the weak-sync traversal boundary.
// Mutations retain the existing serialized runtime/GIL requirement.
class RuntimeHolderState {
public:
    using Holders = HolderList;
    using Iterator = Holders::const_iterator;

    RuntimeHolderState() : sync_cursor_(holders_.sentinel()) {}
    RuntimeHolderState(const RuntimeHolderState&) = delete;
    RuntimeHolderState& operator=(const RuntimeHolderState&) = delete;

    const Holders& holders() const { return holders_; }
    bool contains(const HolderLink& all_link) const { return all_link.linked(); }
    size_t pending_size() const { return pending_holders_.size(); }

    void add(HolderLink& all_link, HolderLink& pending_link, VarHolder* holder, bool is_pending) {
        holders_.push_front(all_link, holder);
        if (is_pending) pending_holders_.push_front(pending_link, holder);
    }

    void mark_pending(HolderLink& all_link, HolderLink& pending_link, VarHolder* holder) {
        if (all_link.linked() && !pending_link.linked())
            pending_holders_.push_front(pending_link, holder);
    }

    void replace(HolderLink& old_all, HolderLink& new_all,
                 HolderLink& old_pending, HolderLink& new_pending,
                 VarHolder* new_owner) {
        if (sync_cursor_ == &old_all) sync_cursor_ = &new_all;
        holders_.replace(old_all, new_all, new_owner);
        pending_holders_.replace(old_pending, new_pending, new_owner);
    }

    void erase(HolderLink& all_link, HolderLink& pending_link) {
        if (all_link.linked()) {
            if (sync_cursor_ == &all_link) sync_cursor_ = sync_cursor_->next;
            holders_.unlink(all_link);
        }
        pending_holders_.unlink(pending_link);
    }

    VarHolder* peek_pending() const {
        if (sync_cursor_ == holders_.front_link()) return nullptr;
        return sync_cursor_->prev->owner;
    }

    // The executor checks the candidate's target cutoff before consuming it.
    void consume_pending() {
        if (sync_cursor_ != holders_.front_link()) sync_cursor_ = sync_cursor_->prev;
    }

    // Visit every potentially-unfinished holder. Returning false from `fn`
    // unlinks that holder from `pending_holders_` (lazy pruning once
    // `holder->var->is_finished()` becomes true).
    template <class Fn>
    void for_each_pending(Fn&& fn) {
        HolderLink* cur = pending_holders_.head_.next;
        while (cur != &pending_holders_.head_) {
            HolderLink* next = cur->next;
            if (!fn(cur->owner)) pending_holders_.unlink(*cur);
            cur = next;
        }
    }

private:
    Holders holders_;
    Holders pending_holders_;
    HolderLink* sync_cursor_;
};

EXTERN_LIB RuntimeHolderState& runtime_holder_state();

} // namespace jittor
