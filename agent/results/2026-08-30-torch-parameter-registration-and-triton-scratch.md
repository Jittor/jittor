# torch 的参数注册语义、Triton 尾部 scratch 参数与向量化归约的浮点顺序

- Status: 四道门全绿（native 739 / CPU Torch 1513 / CUDA+oracle 1809 / nox cuda
  `EXIT=0`，五个子会话 105+6+227+2+227 全过）；vLLM 7B 输出正确
- Last reviewed: 2026-08-30
- Baseline: `f10e9480`
- Owner: Torch compatibility and downstream integration maintainers
- Review when: 参数注册规则、Triton 版本、CUDA 显存查询或 CPU 归约的浮点顺序变化

## 结论

三处 shim 缺口被下游大模型推理暴露出来，各自都超出 vLLM 的范围：

1. **参数注册语义**。Jittor 是"赋值即参数"，torch 只认 `nn.Parameter`。shim 下两套
   约定必须同时成立，判据是**谁在赋值**，不是被赋值对象的类。
2. **Triton 尾部 scratch 参数**。3.7 在 kernel 参数表末尾追加 `global_scratch` 与
   `profile_scratch`；桥接只建模了一个前置指针，安全网因此拦下所有这类 kernel。
3. **`torch.cuda.mem_get_info` 是固定的 64 GiB 桩**。推理框架按它规划显存预算。

另外，`test_cpu_results_are_unchanged` 的失败不是缺陷：CPU 归约现在会被向量化，
浮点求和顺序随之改变。原因已用生成的 kernel 与同旗标的 C++ 复现逐位坐实。

## 一、参数注册：判据是赋值方，不是类

### 现象

vLLM 加载 7B 权重后拒绝启动：

```text
ValueError: Following weights were not initialized from checkpoint:
  {'model.layers.2.self_attn.attn.v_range', ...}
```

### 根因

`vllm/model_executor/layers/attention/attention.py:117` 写的是

```python
layer.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
```

torch 里这只是个普通属性；shim 下 `Module.__setattr__` 把任何公开 Var 赋值都标记成
参数，于是它进了 `named_parameters()`，vLLM 的加载器把它当成"检查点里应该有、却没被
初始化"的权重。

十行即可复现，两侧确实分叉：

| | shim（修前） | 真 PyTorch |
|---|---|---|
| `named_parameters` | `['w', 'plain']` | `['w']` |
| `state_dict` | `['buf', 'plain', 'w']` | `['buf', 'w']` |

影响面远不止 vLLM：优化器会去更新非参数，`state_dict` 会多出键。

### 为什么不能"只认 Parameter"

shim 下 `nn.Linear` 就是 Jittor 的类，权重是普通赋值。改成只认 `nn.Parameter`，
ViT 的 200 个参数会几乎全部消失。

### 第一版规则错在哪

先按**模块类的归属**分流（类定义在 `jittor.` 之外就走 torch 规则）。这条规则让
生态对拍掉了一个权重：

```text
no counterpart for saved weights: ['decoder.embed_positions.weight']
```

`WhisperPositionalEmbedding` 是 transformers 定义的类，但它的 `weight` 是
**Jittor 自己的 `nn.Embedding.__init__`** 用普通赋值声明的。类的归属和赋值代码的
归属是两回事。

### 最终规则

判据取**赋值方所在模块**（`sys._getframe(2)` 的 `__name__`）：

- Jittor 自己的代码赋值 → 参数（Jittor 约定），无论实例是谁的子类；
- 第三方代码赋普通 Var → 记入模块的 `_non_parameter_names`，不进
  `parameters()` / `named_parameters()` / `state_dict()`；
- 值带 `_is_torch_parameter`（`nn.Parameter` 或 `register_parameter`）→ 参数，
  并把该名字从非参数集合里移除；
- **只有首次赋值决定归属**。已经是参数的名字保持是参数，这样 `from_pretrained`
  的 dtype 转换用普通 Var 替换权重时，不会把真权重悄悄踢出优化器；
- 整条规则只在 shim 安装后生效，原生 Jittor 的"赋值即参数"不受影响。

`register_parameter` 也补上了标记：torch 里它就是"显式声明为参数"的调用，而
vLLM 用自己的元类 `__call__` 构造 `ModelWeightParameter`，不会经过 `nn.Parameter`。

验证：

| 场景 | 结果 |
|---|---|
| torch 作者的类 + `nn.Parameter` / 普通张量 | 与 PyTorch 逐项一致 |
| torch 作者的类**继承** Jittor 层（Whisper） | `weight` 保留 |
| Jittor 自己的 `nn.Linear` | `weight`, `bias` 均在 |
| 原生 Jittor（不开 shim） | 赋值即参数，不变 |
| ViT 真实模型 | 两侧同为 200 / 200 |

## 二、Triton 3.7 的尾部 scratch 参数

`_topk_topp_kernel` 报"打包 8 个参数、kernel 声明 10 个"，被桥接的安全网拦下。

Triton 的 nvidia 后端在 `driver.c` 的 "Add scratch objects" 处，把
`global_scratch`、`profile_scratch` 两个指针**追加在参数表末尾**；3.2 两者都没有。
桥接原先假设 scratch 是**前置**的 param 0，且只有一个。

改成由元数据驱动：元数据暴露哪个字段，就追加哪个指针；字段存在但尺寸为 0 时
kernel 仍然声明该参数，此时传空指针（与 Triton 自己的行为一致）。分配大小也补上了
Triton 那份 `grid_size * num_ctas` 的缩放。

顺带确认了一件事:`PassManager::run failed` **与 Jittor 无关**——在真 PyTorch 环境
里把 Triton 降到 3.2.0，同一 kernel 在同一行挂同样的错。

## 三、`mem_get_info` 返回真实显存

原先是 `lambda *a, **k: (64*1024**3, 64*1024**3)`。torch 的语义是 `cudaMemGetInfo`：
整卡的 free/total，包含其它进程、CUDA context 和 Jittor 池里已持有但空闲的块。
vLLM 用 `total*util - (total-free) - 激活峰值` 规划 KV cache，读到虚构的 64 GiB
就会算出无意义的预算。

改为调用 `cudaMemGetInfo`，取不到 cudart 时退回 Jittor 的 `total_cuda_ram`。
（`memory_reserved` 的映射本来就是对的：实测 `total_cuda_used` 跟踪的是**池**，
free 之后不降、`jt.gc()` 之后才降。）

## 四、向量化让 CPU 归约的求和顺序变了（不是缺陷）

`test_cpu_results_are_unchanged` 失败：2^18 个标准正态求和，相对误差 1.5e-4，
超过 1e-4 的容差。

生成的 kernel 是标量顺序累加——accumulator pass 把 `yp[yid]` 提成了局部变量，
去掉了每次迭代的内存往返，编译器因此可以重排。Jittor 用的旗标是 `-Ofast -march=native`，
`-ffast-math` 允许重结合。用同样旗标编译同一个循环：

| 求和方式 | 结果 |
|---|---|
| `-O2` 普通循环（朴素顺序） | 42.199001 |
| `-Ofast -march=native` 普通循环 | **42.191078** |
| 显式 8 路部分和 | **42.191078** |
| float64 参照 | 42.197387 |

与 Jittor 的 42.191078 逐位相同，即 AVX2 的 8 路向量化求和。

这组数据由 209000 的总量抵消到 42（条件数约 5000），固定 `rtol` 考的是数据运气而非
实现——朴素顺序恰好更近，8 路恰好更远。测试改为浮点求和的教科书误差界
`|err| <= log2(n) * eps * sum|x|`，实测 0.0063 远在界内。

## 五、vLLM 7B：十个阻碍全部跨过，输出正确

Qwen2.5-7B-Instruct 现在给出 `' Paris. Which of'`（token `[12095, 13, 15920, 315]`）。

| # | 阻碍 | 归属 |
|---|---|---|
| 5 | `v_range` 被当成参数 | Jittor（已修，见第一节） |
| 6 | `gate_up_proj.weight` KeyError | 第 5 条的回归，已修 |
| 7 | Triton `PassManager::run failed` | 环境（Triton 3.2 太旧） |
| 8 | scratch 参数个数不符 | Jittor（已修，见第二节） |
| 9 | KV cache 预算为负 | 配置（收小 profiling 批量） |
| 10 | 前向输出全 NaN | **第 5 条的第二次回归**，已修 |

### 第 10 条：未初始化的 bias 进了矩阵乘

定位过程是逐层缩小的：

1. 按层量 NaN：第 0 层 attention 输出干净，第 1 层输入干净却输出全 NaN。
2. 把那次调用的输入落盘、在 vLLM 之外原样回放——**离线复现**，说明是数据不是图上下文。
3. 看数据：Q/K/V 的量级是 **1e34**。bf16 装得下，但 `q·k` = 1e68 在 fp32 溢出成
   `inf`，softmax 于是给出 NaN。所以真正的问题在上游。
4. 改量 absmax 而不是 NaN，逐子模块回溯：`L1.input_ln` 输出 24.75（正常），
   `L1.qkv_proj` 输出 **1.038e34**——而它的 weight absmax 是 0.668。
5. 查 bias：`in_params=['weight']`，**bias 根本不在 `named_parameters()` 里**，
   加载器从不填充它，`torch.empty` 的未初始化内存（1e34 / 2.6e36 / 4.3e37）直接
   参与计算。第 0 层碰巧是 0，所以只有第 1 层起才爆。全模型只剩 114 个参数。

根因是第 5 条改动的第二次回归。vLLM 的 linear 写的是
`self.bias = Parameter(torch.empty(...))`，而适配器把 shim 的 `Parameter` 元类
`__call__` **整个替换**成自己的实现，从不设 `_is_torch_parameter`。第 6 条我只补了
`register_parameter` 那条路径，这条直接赋值的路径还漏着。

修在适配器：它既然替换了 shim 的 Parameter 构造函数，就得守住那个契约。
修后参数数 `114 -> 199`，28 个 qkv bias 全部就位，量级 27~171，输出正确。

### 教训

这条规则的失败模式是**静默丢参数**——只有下游数值爆炸才暴露。契约本身是对的
（torch 里普通 Tensor 属性同样不是参数），但任何绕开 `nn.Parameter` 构造参数的
第三方代码都会踩中。凡是替换 shim 构造函数的适配器，都必须保留标记。

过程中还发现基准脚本自己漏了 `bs.patch_vllm()`，适配器的 attention/MoE 补丁因此
全是空转，走的是 vLLM 原版 FlashAttention。

## 复现

```bash
# 参数注册的两侧对拍
JITTOR_TORCH_SHIM=1 python -c "..."   # 见 tests/compat/torch/_torch_compat_checks.py

# 归约的求和顺序
g++ -Ofast -march=native red.cc && ./a.out
```
