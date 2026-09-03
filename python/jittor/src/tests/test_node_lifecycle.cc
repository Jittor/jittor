// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "node.h"

namespace jittor {

struct TestLifecycleObserver final : NodeLifecycleObserver {
    Node* created = nullptr;
    Node* destroyed = nullptr;

    void node_created(Node* node) override { created = node; }
    void node_destroyed(Node* node) override { destroyed = node; }
};

struct TestLifecycleNode : Node {
    TestLifecycleNode() { notify_node_created(this); }
};

struct TestLifecycleObserverGuard {
    NodeLifecycleObserver* previous;
    explicit TestLifecycleObserverGuard(NodeLifecycleObserver* observer)
        : previous(set_node_lifecycle_observer(observer)) {}
    ~TestLifecycleObserverGuard() { set_node_lifecycle_observer(previous); }
};

JIT_TEST(node_lifecycle_observer) {
    TestLifecycleObserver observer;
    TestLifecycleObserverGuard guard(&observer);
    Node* address = nullptr;
    {
        TestLifecycleNode node;
        address = &node;
        ASSERTop(observer.created, ==, address);
        ASSERTop(observer.destroyed, ==, (Node*)nullptr);
    }
    ASSERTop(observer.destroyed, ==, address);
}

} // namespace jittor
