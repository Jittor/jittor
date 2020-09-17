// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "update_queue.h"
#include "executor.h"
#include "node.h"
#include "var.h"
#include "var_holder.h"

namespace jittor {

/*

The update queue is designed to batch update parameters asynchronously. 
It maintains several queues internally. 
Each updated parameter corresponds to a queue, 
and the elements in the queue represent several updates of this parameter. 
When a parameter is updated, 
jittor internally updates the previous parameter several times 
instead of the current parameter.

update queue 设计用于批量异步更新参数，其内部维护了若干个队列，
每一个被更新的参数对应了一个队列，而队列中的元素代表了这个参数
的若干次更新。当一个参数被更新，jittor内部会批量更新若干次之前的
参数，而不是当前参数。

below fig shows a async update process

下图演示了一个异步更新的过程：

first iter
第一次迭代：

      \ iter  0
   param
     a        0
     b        0
     c        0
     d        0
   
second iter
第二次迭代：

      \ iter  0 1
   params
     a        0 1
     b        0 1
     c        0 1
     d        0 1
   
third iter begin
第三次开始时，迭代0的update被执行：
      \ iter  0 1 2
   params
     a       [0]1 2
     b       [0]1
     c       [0]1
     d       [0]1
   
third iter end
第三次结束：

      \ iter  0 1 2
   params
     a          1 2
     b          1 2
     c          1 2
     d          1 2

 update_queue_auto_flush_delay: 异步多少个iter更新.

update queue的提出主要是为了解决统一计算图规模持续增长（lived_var不断变多）的问题，
在 update queue 提出之前， 计算图运行是由optimizer负责的，optim.step被调用的
时候，会自动运行还没有运行的计算图，已经运行的计算图节点会被回收，从而计算图规模可以
在每次迭代之间保持一个常数。

但是如果用户并没有调用optim.step进行更新，计算图就会持续增长，比如下面两种情况：

* 训练 GAN 的时候，只用 SGD 运行了 generator，没有用SGD 运行 discriminator， 
  discriminator 的 batch norm 参数持续不断地更新，但是一直没有运行，导致计算图
  规模持续增长。
* 用户在 inference 的时候忘记设置 model.eval, 这时候因为没有 SGD 刷新参数，
  然后 batch norm 的参数持续不断更新，再次导致计算图规模持续增长。

这些细节对于用户来说过于难以理解（LD：我有时候都很晕），一个粗暴的解决方案是 jt.sync_all,
直接强制刷新全图，把没运行的都运行了，但是这会导致显存占用过大，因为 sync_all 运行的
拓扑顺序不优。

为了让用户可以不关心这些细节， 我们在参数更新的时候，使用 var.update(new_var)，
这个接口会把更新托管给 update queue， 从而不需要关心底层计算图的大小。

 */

DEFINE_FLAG(int, update_queue_auto_flush_delay, 2, "when size of a update queue is great than this value, update queue trigger auto flush(default 2).");

UpdateQueue update_queue;

void UpdateQueue::auto_flush() {
    vector<Var*> vars;
    vars.reserve(queue.size());
    for (auto& l : queue) {
        while (l.size() && l.size() >= update_queue_auto_flush_delay) {
            auto iter = l.end(); iter--;
            auto v = iter->v;
            vars.push_back(v);
            map.erase(v);
            v->flags.set(NodeFlags::_in_update_queue, 0);
            l.pop_back();
        }
    }
    LOGvv << "auto flush var size" << vars.size();
    exe.run_sync(move(vars), false);
}

void UpdateQueue::push(Var* v, Var* prev) {
    if (v->flags.get(NodeFlags::_in_update_queue))
        return;
    list<list<Item>>::iterator owner;

    if (prev->flags.get(NodeFlags::_in_update_queue)) {
        auto iter = map.find(prev);
        ASSERT(iter != map.end());
        owner = iter->second->owner;
    } else {
        queue.emplace_front();
        owner = queue.begin();
    }
    if (owner->size() >= update_queue_auto_flush_delay) {
        auto_flush();
    }
    v->flags.set(NodeFlags::_in_update_queue);
    owner->emplace_front(UpdateQueue::Item{owner, v});
    map[v] = owner->begin();
    // if total size of update queue is too big,
    // force sync all
    if (map.size() > 100000)
        sync_all();
}

void UpdateQueue::pop(Var* v) {
    auto iter = map.find(v);
    iter->second->owner->erase(iter->second);
    map.erase(iter);
}

} // jittor

