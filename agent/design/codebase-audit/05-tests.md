# 测试体系与质量保障

**核心判断**：测试资产与门禁被彻底解耦。仓库有 289 个测试文件、2513 个测试函数，但任何
CI workflow 能触达的只有 74 个文件（25.6%），PR 门禁只有 50 个（17%）；唯一能跑全树的
入口 `tools/run_test_suite.py` 不被任何 nox session 或 workflow 调用，且它自己一次要跑
4 小时并跳过 35% 的用例。其次投入方向倒挂：跑得最全的 `tests/structure`（234 用例 8071 行）
断言的是模块路径、re-export 恒等和文件行数预算，而算子级反向正确性（gradcheck，覆盖
227 个 OpInfo）因为 `JITTOR_TEST_DEVICES` 与 `@onlyCPU` 的交互，在**所有门禁里都被实例化
为零个用例**。第三，门禁里存在成规模的假绿。最后，`make_tensor` 用进程级递增计数器做
随机种子，用例数据随选择集合变化，失败不可稳定复现。

## 门禁范围
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 门禁是手写白名单，与测试树增长完全脱钩 | `noxfile.py:152` CPU_TESTS 19 条、`:173` oracle 9 条、`:184` CUDA 6 条、`:272` NPU 8 条、`:282` ROCM 1 条。展开后 CI 可达 74/289 文件，215 个文件不在任何 workflow 路径上 | 新增测试默认不进门禁，写了等于没写 | 门禁改为目录加显式排除清单，排除须写理由 | 关键 |
| 未覆盖测试按目录分布集中在核心 | 未覆盖：compat/torch 55、compiler 45、core 32、ops 30、nn 19、data 7、distributed 8、compat/vllm 3、backends/triton 3、backends/cpu 2 | JIT 优化 pass（src/opt 82 文件 7589 行）、内存分配器（src/mem 26 文件）、oneDNN CPU 卷积、dataset/transform（3355 行）全部无门禁保护 | 至少把 compiler+core+ops 全目录纳入 CPU 门禁（不需硬件） | 关键 |
| tools/run_test_suite.py 是文档宣称的完整入口但无人调用 | grep 无命中于 noxfile.py 与 .github/；只出现在 docs 与 agent/results | 唯一的全量口径是人工命令，绿不绿取决于谁记得跑 | 拆成 nox -s full 周期性调度并入 CI，否则删掉并承认没有全量口径 | 关键 |
| 默认 nox 不含任何数值测试 | `noxfile.py:294` sessions = [lint, format, typing, structure, packaging, py37, py312, py313] | 本地 nox 全绿不等于代码能算对 | 把 cpu 加入默认或把默认改名为 static | 主要 |
| optional/rocm/mpi/nccl 四个 session 从不在 CI 运行 | workflows 只调用 cpu、cuda、npu、structure、packaging、py 系列、docs | ROCm 后端、FSDP2/NCCL、mmcv/peft/flash-attn 兼容层无自动验证，而文档把它们列为可复现门禁 | 排期上 runner 或在文档标注手动 | 主要 |
| CUDA 门禁不在 pull_request 上触发 | `.github/workflows/cuda.yml:3` on: push/schedule/workflow_dispatch | fork 提交的 PR 永远没有 GPU 验证 | 对 labeled PR 触发或明确 merge 策略 | 主要 |

## 跳过条件与假绿
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| CPU 门禁条目 tests/nn/test_attention.py 在门禁里 100% 恒跳过 | 该文件只有 1 个用例，受 `:17 skip_this_test = not modules_available("torch","fairseq")` 控制；而 `noxfile.py:1277` 在运行前强制 `REAL_TORCH_SITE=""`，venv 也不装 torch/fairseq | 一条门禁条目常年报告 1 skipped 并被当作通过 | 门禁断言每个条目至少执行 1 个非 skip 用例 | 主要 |
| CPU 门禁条目 test_opt_state_dict.py 函数体就是 return | `tests/optim/test_opt_state_dict.py:11-13` | 门禁里一条恒绿的空条目 | 删除或补真正断言 | 主要 |
| 6 个用例被首行 return 静默停用并报告为 PASS | `backends/cuda/test_bf16.py:223`（注释写 this test cannot pass now）、`test_cudnn_op.py:132`、`core/test_core.py:276/295`、`compiler/test_ring_buffer2.py:97`、`optim/test_opt_state_dict.py:11` | 已知缺陷伪装成通过；xfail_strict 机制被绕开 | 改 expectedFailure 并登记；加禁止首行 return 的静态检查 | 主要 |
| 4 个用例 skipIf(True) 永久停用，其中 2 个是仅有的内存泄漏测试 | `core/test_function.py:291`、`compiler/test_numpy_code_op.py:152`、`ops/test_arg_pool_op.py:110`、`data/test_emnist.py:19` | **内存释放契约零覆盖**；liveness_info 只在 12 个文件零星使用 | 泄漏测试改短循环加 RSS 阈值放 nightly | 主要 |
| 静态可判定：487 个用例（19.5%）依赖加速器，205 个（8.2%）依赖真 PyTorch | AST 统计 2502 个测试方法的 skip 装饰器。实测印证：全量 CPU 双进程 `2359 passed, 1274 skipped`——**35% 的用例在 CPU 机器上从不执行** | 按 skip 原因分桶统计并在 CI summary 输出 | 对"本环境应能跑却 skip"的桶设阈值 | 主要 |
| 真 PyTorch oracle 只有 27/205 个用例进门禁且门禁 fail-open | oracle 9 个文件共 27 个静态用例；`JITTOR_REQUIRE_REAL_TORCH=1` 只在 `noxfile.py:1269-1271` 校验环境变量存在，不校验实际执行了用例 | torch 装坏则 modules_available 返回 False，27 个静默 skip，CI 绿 | require_real_torch 时断言 skipped==0 | 主要 |
| 生态对拍（唯一的真 PyTorch 模型级比对）不可从任何门禁到达 | `test_ecosystem_parity.py:25` skipUnless(REAL_TORCH_PYTHON)；该变量在 noxfile 与 workflows 中零出现；两个文件也不在任何 *_TESTS 里 | **项目核心目标（数值一致、速度 ≤1.07×）完全没有自动门禁**；speed 测试明确说除非设 SPEED_RATIO 否则只测量不断言，而该变量从未被设置 | 建装有真 torch 的 runner 跑 nightly parity | 关键 |
| expect_error 只判断抛了任意异常 | `tests/_helpers/assertions.py:4-9` `except Exception: return`，61 处使用 | 测试自身的 typo 也算通过 | 改签名带 exc_type 与 match | 次要 |
| 错误路径覆盖整体偏薄 | 全树 assertRaises 113 加 expect_error 61 = 174 处，对应 2513 个用例（约 7%） | 越界、dtype 不匹配、形状不合法等最常撞到的路径基本靠运气 | OpInfo 增加 error_inputs_func | 主要 |

## 算子证据链
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| **gradcheck/gradgradcheck 在所有门禁中实例化为 0 个用例** | `tests/ops/test_ops.py:82,111` 的 @onlyCPU；`_helpers/device_types.py:57-65` 用 get_all_device_types 过滤，而 `_helpers/common.py:50-53` 在 JITTOR_TEST_DEVICES 存在时把 cpu 剔除。CUDA 门禁 `noxfile.py:1449` 设 JITTOR_TEST_DEVICES=cuda，bases 只剩 cuda，`device_types.py:219` 的 restriction 检查跳过所有 @onlyCPU 方法，TestGradientsCUDA 是空类。且 test_ops.py 根本不在 CPU_TESTS 里。实测印证：CUDA OpInfo 221 passed 恰好等于 227 个 OpInfo 的 test_reference 数，反向一个没有 | **227 个算子的导数公式正确性在 CI 中零验证**；文档宣称的第二层证据是空的 | TestGradients 改用 only_for=("cpu",) 显式实例化；把 JITTOR_TEST_DEVICES=cpu 的 test_ops.py 加进 CPU 门禁 | 关键 |
| 空类不报错：设备过滤后生成 0 个方法仍然通过 | `device_types.py:198-232`：bases 为空或方法全被过滤时 scope[cls_name] 是无测试方法的类，pytest 正常收集并报 0 项 | 任何设备过滤配置错误都表现为静默少跑；`JITTOR_TEST_DEVICES=rocm`（`noxfile.py:1505`）与 =mpi（`:1525`）在 get_all_device_types 里直接返回**空列表**，因为它只认 cpu/cuda/npu | 生成器在 bases 为空或方法数为 0 时 raise；两处设备名枚举统一 | 关键 |
| 前向只测一个 dtype，OpInfo.dtypes 声明基本是装饰 | `test_ops.py:59` `@ops(op_db, dtypes=OpDTypes.any_one)`；`device_types.py:83-88` 按 float32→float64→int64 取第一个 | dtypes=all_types_and(...) 给人全 dtype 覆盖的错觉，实际只跑 1 个 | 对 Unary/Binary/Reduction 改成 OpDTypes.supported | 主要 |
| bfloat16 在 harness 层被静默替换成 float16 | `_helpers/common.py:216-217` `if dtype == bfloat16: v = v.float16()` | 任何声明 bf16 的用例实际测的是 fp16，指数位宽差异这一最关键语义差别永远测不到；bf16 的容差设定成了摆设 | 用原生 bf16 构造或从 dtype 组移除并声明未覆盖 | 主要 |
| xfail() 定义了但零使用，已知缺陷一律用 skip() | `tests/opinfo/core.py:96` 定义 xfail；definitions/ 下 `xfail(` 命中 **0**，`skip(` 命中 9 | 文档承诺的"修复会产生 XPASS 并强制清理 ledger"事实上不存在；split/chunk 的 gradcheck、digamma、cholesky 都以"别处已验证"为由跳过而无法验证 | 已复现缺陷改 xfail；skip reason 必须写出那个测试的 nodeid | 主要 |
| OpInfo 覆盖 200 个算子，公开面约 536 个符号 | 227 个 OpInfo 实例 / 200 个 distinct name；`__init__.pyi` 有 536 个 def | 未进注册表的算子（setitem/index_put/nonzero/unique/bincount/einsum/ctc_loss/rms_norm/rope/paged_attention/fused_moe/conv_transpose3d）没有统一三层证据 | 生成公开 API 与 OpInfo 差集作为 structure 门禁一项 | 主要 |
| 63 个 OpInfo 声明 supports_autograd=False（28%） | definitions/ 下 63 处；其中 `fft.py:163-167` 的 fft/ifft/rfft 在 PyTorch 中可导 | 可导算子被标为不可导直接绕过唯一的反向门禁 | 每个该声明要求指向 KI 编号或数学理由 | 主要 |
| gradcheck 本身（221 行）没有自测 | `_helpers/gradcheck.py` 无任何负向验证；仅两处消费 | 227 个算子反向证据的单点：若它因某分支恒返回 True，整层证据无声塌陷 | 加"故意写错导数应当失败"的负向测试 | 主要 |

## 对拍口径与容差
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 设备对拍只跑 float32，而它的动机全是窄整数 bug | `test_device_parity.py:192` sample_inputs("cpu","float32")；文件头 `:11-14` 却把 int8/int16 reduce miscompiles 列为主要动机 | 动机里的 bug 类别恰好测不到 | 增加 dtype 轴 | 主要 |
| 容差是全算子一刀切的绝对常数，与规模无关 | `:159-163` FWD_TOL=2e-4 / GRAD_TOL=2e-3 / FWD_PE_TOL=2e-3 / GRAD_PE_TOL=5e-3 / PE_ATOL=1e-3 | 大规约的合法重排误差可能超阈值，靠调大常数放松所有算子 | 容差按 sqrt(reduce_size)*eps 缩放或改 per-OpInfo | 主要 |
| linalg 7 个算子的 parity 靠 cupy 探针，探针失败即整组静默跳过 | `:59 _LINALG_OPS`、`:63-89 _cuda_linalg_works()`、`:190-191 skipTest` | 换 CUDA 版本或 cupy 装坏则 det/inv/svd/qr/cholesky/solve/slogdet 的设备一致性全部无声消失 | 探针失败应为 error | 次要 |
| 生态对拍容差分两档但都不在门禁内 | `_ecosystem_harness.py:287-288` 2e-3/1e-2；`test_ecosystem_speed.py:60-62` 大模型放宽到 2e-2/5e-2 | 12 层模型的 5e-2 反向容差能吞掉真实的层级实现错误 | 先让 parity 进 nightly 再讨论容差 | 主要 |
| CPU 作为 parity oracle 的前提当前不成立 | 文档明写"CPU 只有在被前两层独立 pin 住之后才是可用的 oracle"；但第 1 层在 CUDA 门禁下只跑 cuda，第 2 层不跑 | 三层证据链的第 3 层建立在未验证的第 1、2 层之上 | 修好 gradcheck 入门禁后 parity 结论才有意义 | 关键 |

## 测试自身的可靠性
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| make_tensor 默认种子来自**进程级递增计数器** | `_helpers/common.py:146` `_seed_counter = itertools.count(0x5EED)`；`:184` 用 next(_seed_counter) | 同一用例在全量跑和 -k 单跑下拿到**不同输入数据**：全量失败的 case 单跑复现不了；插入或删除任何用例都会平移下游所有数据；无法用 xdist 或随机顺序 | 种子改为 hash(nodeid+shape+dtype) 的确定性函数并在失败信息里打印 | 关键 |
| 14 处在 setUp 或用例体里改进程级 flag 且所在类无 tearDown | `core/test_complex64_linalg.py:50`、`compat/torch/test_torch_hf_cuda_device.py:95`、`test_ecosystem_device_selection.py:61/64/69/77`、`nn/test_bmm.py:21`、`nn/test_loss3d.py:80`、`core/test_misc_op.py:281`、`ops/test_linalg.py:309`、`backends/rocm/test_rocm.py:330` | 一个类跑过之后同进程后续用例的 use_cuda/use_rocm 被改写，后续用例可能在错误设备上通过 | 统一走 flag_scope；加静态检查禁止裸赋值 | 主要 |
| conftest 用 sys.argv 嗅探决定进程语义 | `tests/conftest.py:24-60`，`:43 SELECTION_IS_BROAD`；`:174 pytest_ignore_collect` 在 native 会话整体忽略 TORCH_MODE_PATHS | 语义随调用方式改变；用 -k、xdist worker 或 IDE runner 时行为不可预期。CPU 门禁的 --collect-only 因此**连 62 个 Torch-mode 文件的可导入性都没检查** | 模式由显式环境变量决定 | 主要 |
| retry 装饰器吞掉任意异常 | `_helpers/retry.py:9-16` `except Exception: pass` | 不稳定的 tuner 路径被掩盖成绿，不稳定率不可观测 | 记录并上报重试次数 | 次要 |
| 门禁在非默认配置下运行 | 运行时默认 use_parallel_op_compiler=16（`parallel_compiler.cc:27`）；但 `noxfile.py:687/1128/1548`、`tools/run_test_suite.py:59`、`test_notebooks.py:253` 全部强制 0，`test_device_parity.py:172` 也在 setUpClass 关掉 | 用户拿到的默认路径在全量套件里从不被执行 | nightly 保留一条并行编译的全量运行 | 主要 |
| _session_env 直接继承宿主环境 | `noxfile.py:363` `env = os.environ.copy()` | OMP_NUM_THREADS、OMP_PROC_BIND、CPU 亲和性、MKL_* 全部泄漏进门禁 | 门禁显式设定并断言线程数与亲和掩码 | 主要 |

## tests/structure 的成本
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 结构门禁规模超过数值门禁两倍 | structure 22 文件 8071 行 234 用例全部在 PR 门禁；CPU 数值门禁 16+9 文件 142 个静态用例 | 每次重构 nn 的物理布局都要改 1912 行测试；而改错一个 kernel 的算术不会被任何 PR 门禁挡住 | 压缩成公开 API 快照、循环依赖检查、打包内容三类 | 主要 |
| 用文件行数当架构契约 | `test_nn_structure.py:1812-1821` facade ≤300 行、实现模块 ≤350 行 | 加一段必要注释就会让门禁变红，鼓励把逻辑拆到别处而不是写清楚 | 改软告警或按公开符号数 | 次要 |
| 一次性迁移守卫被永久固化 | `test_cleanup_structure.py:24-47` 列举 21 个已删除路径断言不存在 | 迁移完成后永远为真，纯粹门禁负重；这类测试在 22 个文件里成规模存在（62 处 subprocess 进一步拉高耗时） | 迁移守卫设过期时间 | 次要 |
| marker 体系基本是死代码 | pyproject 注册 9 个 marker；`conftest.py:113-150` 给每个用例打设备 marker，但 `noxfile.py:410` 从不传 -m；slow 只打给一个文件且无人按它筛选 | pyproject 里描述的"快速 PR 门禁"层根本不存在 | 真正建立 -m "not slow" 的快门禁，或删掉 marker | 主要 |

## 完全没有测试保护的关键契约
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| C++ 单元测试全部不在门禁 | `src/test/` 9 个文件 1033 行（expr/kernel_ir/op_compiler/op_relay/sfrl_allocator/setitem_op/jit_key/nano_vector/fast_shared_ptr），桥接文件 `tests/compiler/test_jit_tests.py` 不在任何 *_TESTS 里 | 表达式求解、kernel IR、op 编译器、SFRL 分配器的单元级证据零执行 | 加进 CPU 门禁（成本极低） | 关键 |
| 该桥接自身会静默产生 0 个用例 | `test_jit_tests.py:31-36` 遍历 dir(jt.tests)，为空时无任何方法，pytest 报 0 项通过 | wheel 裁掉 src 或扫描失败时测试通过但什么也没跑 | 断言 len(names) > 0 | 主要 |
| 安装后 wheel 的验证只有一次乘法 | `selftest.py` 共 60 行只验证 `[1,2,3]**2` 的前向与梯度 | 打包遗漏任何模块都不会被发现 | 扩成 conv+bn+optimizer 三步训练加关键子包 import 清单 | 主要 |
| CPU 卷积后端（oneDNN/MKL）无门禁 | `tests/backends/cpu/` 2 文件 236 行不在任何 session | 默认 CPU 卷积路径没有自动验证 | 加入 CPU 门禁 | 主要 |
| 服务化推理关键算子无门禁 | test_paged_attention、test_fused_moe、test_serving_ops、test_rnn、test_norm、test_linear 等 19 个 tests/nn 文件不在任何 session | vLLM 适配依赖的 paged attention 与 fused MoE 无回归保护 | CPU 可跑部分进 CPU 门禁 | 主要 |
| dataset/transform 管线无门禁 | tests/data 7 文件 67 用例全部不在 session；对应源码 dataset 1423 行加 transform 1932 行 | 数据管线任何回归都要等用户报 | 至少两个主文件进 CPU 门禁 | 主要 |
| notebook 门禁只验证不抛异常且近半代码单元被跳过 | `test_notebooks.py:236` 单个用例 timeout 1800 跑 12 个 topic；examples/notebooks/ 共 34 个 skip-execution 标签；守卫只要求每 topic 至少 1 个可执行单元 | 教程数值结论无验证；一个 topic 即使 90% 单元被跳过也算过 | 按 topic 参数化；skip 标签需写理由并设比例上限 | 次要 |

## 耗时分布
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| CUDA 门禁 80% 时间花在一个文件上 | CUDA backend 目录 23:51、dtype coverage 7:33、**device parity 2:39:10**、TF32 3s、strict OpInfo 9:50 | parity 221 用例平均 43 秒每个，而同样 221 用例的 OpInfo 只要 2.7 秒每个，16 倍差距 | 根因是 `test_device_parity.py:172` 在 setUpClass 把 use_parallel_op_compiler 设为 0 导致约 450 个算子变体串行编译；改按算子分片加并行进程 | 主要 |
| 全量 CPU 套件 4 小时且不分层 | native 1:45:28、torch 2:21:51、structure 2:31 | 没有人会在改一行代码后跑它，退化成每月一次的人工快照 | 建 smoke（<5 分钟进 PR）与 full（nightly）两层 | 主要 |
| 门禁每个条目起一个独立 pytest 进程 | `noxfile.py:404-423`，CPU 门禁因此起 28 个进程 | 每个进程重付 import 成本；进程隔离的收益本应由测试自身的 flag_scope 纪律提供 | 修好 flag 泄漏后合并 | 次要 |
