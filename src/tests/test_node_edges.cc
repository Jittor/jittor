// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "node.h"

namespace jittor {

struct EdgeTestNode : Node {};

struct CountedValue {
    static int alive;
    int value;
    explicit CountedValue(int value=0) : value(value) { ++alive; }
    CountedValue(const CountedValue& other) : value(other.value) { ++alive; }
    CountedValue(CountedValue&& other) noexcept : value(other.value) { ++alive; }
    CountedValue& operator=(CountedValue&& other) noexcept {
        value = other.value;
        return *this;
    }
    ~CountedValue() { --alive; }
};

int CountedValue::alive = 0;

JIT_TEST(small_vector_storage_contract) {
    {
        SmallVector<CountedValue, 2> values;
        CHECK(values.using_inline_storage());
        values.emplace_back(1);
        values.emplace_back(2);
        CHECK(values.using_inline_storage());
        values.emplace_back(3);
        CHECK(!values.using_inline_storage());
        CHECKop(values.capacity(),>=,3u);
        ASSERTop(CountedValue::alive,==,3);

        values.erase(values.begin() + 1);
        ASSERTop(values.size(),==,2u);
        ASSERTop(values[0].value,==,1);
        ASSERTop(values[1].value,==,3);
        ASSERTop(CountedValue::alive,==,2);

        SmallVector<CountedValue, 2> moved(std::move(values));
        ASSERTop(values.size(),==,0u);
        ASSERTop(moved.size(),==,2u);
        ASSERTop(moved[0].value,==,1);
        ASSERTop(moved[1].value,==,3);
        ASSERTop(CountedValue::alive,==,2);
    }
    ASSERTop(CountedValue::alive,==,0);
}

JIT_TEST(node_edge_table_contract) {
    EdgeTestNode producer0, producer1, producer2;
    EdgeTestNode consumer0, consumer1, consumer2;

    consumer0.add_inputs(vector<Node*>{&producer0, &producer1, &producer2});
    ASSERTop(consumer0.input(0),==,(Node*)&producer0);
    ASSERTop(consumer0.input(1),==,(Node*)&producer1);
    ASSERTop(consumer0.input(2),==,(Node*)&producer2);
    for (uint i = 0; i < consumer0._inputs.size(); ++i) {
        auto& input = consumer0._inputs[i];
        ASSERTop(input.reverse().node,==,(Node*)&consumer0);
        ASSERTop(input.reverse().back_index,==,i);
    }

    consumer1.add_inputs(vector<Node*>{&producer0});
    consumer2.add_inputs(vector<Node*>{&producer0});
    ASSERTop(producer0.output(0),==,(Node*)&consumer0);
    ASSERTop(producer0.output(1),==,(Node*)&consumer1);
    ASSERTop(producer0.output(2),==,(Node*)&consumer2);

    // Removing the middle consumer preserves creation order and repairs the
    // shifted edge's reverse index.
    consumer1.release_inputs();
    ASSERTop(producer0._outputs.size(),==,2u);
    ASSERTop(producer0.output(0),==,(Node*)&consumer0);
    ASSERTop(producer0.output(1),==,(Node*)&consumer2);
    ASSERTop(consumer2._inputs[0].back_index,==,1u);
    ASSERTop(consumer2._inputs[0].reverse().node,==,(Node*)&consumer2);

    consumer0.release_inputs();
    ASSERTop(producer0._outputs.size(),==,1u);
    ASSERTop(producer0.output(0),==,(Node*)&consumer2);
    ASSERTop(consumer2._inputs[0].back_index,==,0u);
    consumer2.release_inputs();
    CHECK(producer0._outputs.empty());
}

} // namespace jittor
