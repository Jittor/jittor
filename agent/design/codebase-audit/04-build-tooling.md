# 构建、打包、缓存与开发工具链

**核心判断**：`import jittor` 不是导入一个库，而是启动一套完整的构建系统：探测编译器与
驱动、拉取第三方二进制、生成代码、编译整个 C++ 内核、再逐个编译算子。这条路径上每一步
都写在模块顶层，没有阶段划分、没有事务、没有"探测结果"这层抽象——探测结果不落盘，
失败只能靠 LOGf 中止进程。缓存被切成两半：**目录**按工具链分区（编译器、Python、主机名、
CPU 型号、git 分支），**内容**按命令行加源码哈希判定，中间那层"这次构建用的是哪套配置"
（nvcc_flags、cuda_archs、enable_lto、shim 的严格数学开关）两边都没有，于是不同配置的
进程在同一目录里互相重编互相覆盖。锁有两把——Python 用 flock、C++ 用 fcntl record lock
——在 Linux 上互不排斥，而两边都以为自己拿到了同一把 jittor.lock。第三方依赖分散在三个
文件里的 URL/MD5 三元组，进程级关闭了 TLS 校验，门禁每跑一次就重新从清华的一台主机下载
一遍。门禁本身是手工维护的测试白名单：289 个测试文件里只有 86 个能被任一 session 触及。

## 首次导入做了多少事
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 探测结果一次都不缓存，每次 import 重跑十几个子进程 | `g++ --version`（`jittor_utils/__init__.py:757`）、`git branch`（`:504`）、`python3-config`（`:660/701`）、`nvidia-smi -q -u`（`install_cuda.py:62`）、`nvcc --version`（`compiler.py:1024/1044/1465`、`compile_extern.py:195/295` 每库一次共 6 次）、`gdb`/`addr2line`（`compiler.py:1037-1038`）、`query_cuda_cc` 整个解释器（`compiler.py:1051`）、`mpicc --showme` ×3、`hipcc --version` | 热缓存下 import 仍有几百毫秒到数秒纯探测开销；每个探测都是可失败点；自死锁正来自其中之一 | 探测结果写 `cache_path/probe.json` 带工具链 mtime 失效 | 主要 |
| 导入期无条件 import torch | `compile_extern.py:903-914`，FIX_TORCH_ERROR 默认 1，异常被 `except: pass` 吞掉 | 装了 torch 的环境每次 import jittor 多付 2–5 秒和上百 MB；`dirty_fix_pytorch_runtime_error` 还改 `os.RTLD_GLOBAL`（`jittor_utils/__init__.py:738`）污染 stdlib 常量 | 改惰性；不要修改 os 模块常量 | 主要 |
| 导入期可能自动下载约 2GB CUDA 工具链且不询问 | `compiler.py:1029-1030` → `install_cuda.py:187` 从清华镜像下 `cuda12.2_cudnn8_linux.tgz`；触发条件只是有驱动但 PATH 无 nvcc | 用户在 import 处静默挂起十几分钟；受限网络下表现为无响应 | 默认关闭改显式命令；至少先打印大小并要求确认 | 主要 |
| 导入期会 os.execl 重启用户进程 | `install_cuda.py:113-122` 读 `/proc/self/cmdline` 后 `os.execl(sys.executable, sys.executable, *argv[1:])`，整段裹在 `except: pass` | shebang 启动的脚本 argv[0] 是脚本本身，`argv[1:]` 丢掉脚本 → 重启成 `python arg1`；import 之前的进程状态全丢；在 MPI rank 或 multiprocessing worker 中是灾难 | 库的 import 里不得 exec 自己；用 dlopen 绝对路径代替改 LD_LIBRARY_PATH | 关键 |
| "请重新运行你的命令"用退出码 0 | `compiler.py:926-928` LOG.e 后 `sys.exit(0)` | 改了 `src/utils/*.cc` 后 `python train.py` 什么都没做就成功退出，CI 看到的是成功 | 非零退出码 | 主要 |
| 依赖缺失一律 LOGf 中止进程而非抛异常 | `compile_extern.py:263`、`:42` search_file 找不到即 LOG.f；`compiler.py:942` 用裸 assert | 用户拿到 abort 加栈而不是可捕获异常；`python -O` 下裸 assert 直接消失 | 构建期失败一律抛带上下文的 RuntimeError | 主要 |

## 缓存键与缓存布局
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 编译配置不进缓存路径，不同配置的进程在同一目录互相重编 | 缓存路径成分见 `jittor_utils/__init__.py:480-525`：版本、cc 版本、py 版本、platform+node、CPU 型号、`__file__` 哈希前 4 位、git 分支。**不含** nvcc_flags/cc_flags/cuda_archs/enable_lto/kernel_flags。而 torch shim 会往 nvcc_flags 塞 `--fmad=false --prec-div=true --prec-sqrt=true` 并去掉 `--use_fast_math`（`compat/shim/preflight.py:179-189`） | 同一台机上 shim 开与关共用 `cache_path/jit/`，每次切换重编全部 CUDA kernel；并发时后写者替换前者已 dlopen 的 .so 产生同一库两份映射——这个失败模式 `compiler.py:929-938` 的注释已为 jit_utils_core 单独描述过但没推广 | 缓存路径追加构建配置指纹；写新目录切指针不要原地重建 | 关键 |
| helper_cuda.h 被显式排除在缓存键之外（**2026-09-03 更正**：本行「删除这条例外」的建议**按字面做会让 CPU 构建整个失败**——47 处 include 都在 `#ifdef HAS_CUDA` 里，而 `extern/cuda/inc` 只进 `nvcc_flags`，手写扫描器解析不到就 `ASSERT(found)`。这条必须与下一行的 `-MD -MF` 同一提交完成，不能单独做） | `src/utils/cache_compile.cc:184`：`if (inc != "test.h" && inc != "helper_cuda.h")` 才计入依赖；该头被 47 个源文件包含 | 改 `extern/cuda/inc/helper_cuda.h` 不触发任何重编译，直接用过期目标文件 | **更正（2026-09-03，构建分区实测）：不能只"删除这条例外"。** 这两条例外是扫描器不认识 `#ifdef` 的补丁：47 个文件里的 `#include "helper_cuda.h"` 都在 `#ifdef HAS_CUDA` 里，而 `extern/cuda/inc` 只加进 `nvcc_flags`（`compiler.py:1494`），不在 CPU 编译的 `-I` 列表里。裸删这条例外之后，**CPU 构建**扫到这一行会解析不到文件并触发 `cache_compile.cc:352` 的 `ASSERT(found)` 而整个失败。`test.h` 同理（只在 `#ifdef TEST` 下包含，且在 `src/utils/` 不在 `src/`）。所以这一条必须和同表的 `-MD -MF` 一起做——让编译器回答依赖，就同时消掉了 `#ifdef` 与`<...>` 两个问题，也就不再需要任何按文件名写死的例外 | 主要 |
| 只跟踪 `#include "..."`，不跟踪 `<...>` | `cache_compile.cc:176` 只在 `src[k]=='"'` 时记录 | 尖括号包含的项目内头文件改动不触发重编 | 依赖跟踪改用编译器的 `-MD -MF`，不要手写预处理器 | 主要 |
| 缓存内容哈希是可构造碰撞的多项式哈希 | `src/misc/hash.h:31-37`：`v += mul*c; mul *= 257` 模 2^64 线性 | 判定产物是否最新的唯一依据比同仓库已在用的 md5 还弱 | 换 SHA-256 或 xxhash64 | 次要 |
| 缓存路径含主机名却不含 CPU 指令集边界 | `jittor_utils/__init__.py:488` 用 `platform.node()`；`compiler.py:1094` 无条件加 `-march=native`；CPU 只以 model name 前 14 字符加 2 位哈希入键 | 主机名入键使集群每节点全量重编，缓存无法共享；反过来同型号不同微码的机器共享缓存可能拿到非法指令 | 主机名移出键；用编译器实际展开的 -march=native 结果入键 | 主要 |
| 缓存路径依赖 git branch | `jittor_utils/__init__.py:503-514` 在 jittor_utils 目录跑 git branch 取 `* ` 开头那行，失败静默回落 default，结果写回 `os.environ["cache_name"]`（`:519`）传给所有子进程 | 切分支等于全量重编；detached HEAD 时 for/else 拿到最后一行得到错误 cache_name；pip 装到某 git 仓库内的 site-packages 会意外继承那个仓库的分支名 | 缓存名由源码内容哈希决定，不要把内部状态写回 os.environ | 主要 |
| 项目路径只用 4 位十六进制入键 | `jittor_utils/__init__.py:495` `get_str_hash(__file__)[:4]` | 65536 空间，两个并行 worktree 撞键就是两套源码共用一个缓存目录 | 至少 12 位或直接用规范化绝对路径 | 次要 |
| clean_cache 与实际布局脱节 | `jittor_utils/clean_cache.py:26-27` 删 `cache_path/default`、`/master`（当前布局下分支名是第九级）；`clean_core` 的 `jt*` 通配同时匹配 `jtcuda`；`clean_swap` 删 `<root>/tmp` 而 swap 文件在构建树**里面**的 `tmp`；cutlass、mkl、msvc、auto_diff、probe.json 任何子命令都够不到 | 清缓存清不干净，且"清编译产物"会顺手删掉自带的 CUDA 工具链 | 清理逻辑从同一份布局定义生成（**已完成**：`jittor_utils.CACHE_GROUPS` / `cache_group_paths()`） | 次要 |

## 锁与并发
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| Python 与 C++ 用的是两套互不排斥的锁 | Python `jittor_utils/lock.py:30` 用 `fcntl.flock`（BSD）；C++ `src/lock.cc:47-53` 用 `fcntl(F_SETLKW)`（POSIX record lock）。Linux 上两类锁完全独立 | Python 侧（下载、compile_custom_ops）与 C++ 侧（`parallel_compiler.cc:241`）都以为自己独占 jittor.lock，实际可同时进入；这是"缓存里出现散布失败与段错误"的一种合理解释 | 统一到一种锁 | 关键 |
| POSIX record lock 的"关闭任一 fd 即释放"陷阱 | `lock.py:22` 打开一次（顺带截断锁文件），`src/lock.cc:34` 对同一路径再打开一次；POSIX 语义下同一进程关闭任一 fd 释放该文件全部 record lock | Python 侧任何一次析构或重新 import 都可能悄悄放掉 C++ 侧的锁 | 单一 fd 单一锁类型 | 主要 |
| 全局锁被长任务持有 | `jittor_utils/misc.py:49` 用 `@lock.lock_scope()` 装饰 `download_url_to_local`，整个几百 MB 下载期间持锁；`compile_extern.py:616` 的 NCCL `make -j8` 在锁域内 | 机器上任何一个 jittor 进程下载或编译时其余全部阻塞（观测到 40 分钟卡死） | 锁粒度按产物路径切分；下载用 .part 加原子 rename | 主要 |
| disable_lock=1 是无保护的逃生阀 | `lock.py:18`、`src/lock.cc:31` | 用它绕开卡死会得到多进程同时写同一 .so 的静默损坏 | 启用时明确告警并纳入缓存指纹 | 次要 |
| 并行算子编译器的线程数是 RAM 的函数且只算一次 | `parallel_compiler.cc:218` `static int thread_num = max(1, min(use_parallel_op_compiler, total_cpu_ram/3GB))` | 4 核 64GB 容器里开 16 个编译线程；运行时改 flag 不会改变线程数（`create_threads` 见 `:107-108` 早退） | 按 sched_getaffinity 与 cgroup 配额取；去掉 static | 主要 |
| 并行编译器用 volatile 加自旋计数器代替同步原语 | `parallel_compiler.cc:226` `static volatile int has_error`；error_msg 是无锁写的 static string；主线程屏障是 `while (prev_i < n)` 自旋（`:327-348`）无超时；`wait_all()` 只等 5 秒就打印 "Compile thread timeout, ignored."（`:117-128`） | 工作线程异常退出即主线程永久自旋；volatile 不是原子，数据竞争是 UB——这正是"求稳所以门禁全关"的根因 | 用 atomic 加 future/join，错误用 exception_ptr 传递；修好后默认打开否则删掉 | 主要 |

## 第三方二进制下载
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 进程级关闭 TLS 证书校验 | `jittor_utils/__init__.py:765-769` `ssl._create_default_https_context = ssl._create_unverified_context` 无条件无开关；`compiler.py:1398-1400` 对 sw_64 再来一次 | 影响的不只是 jittor 自己的下载，而是同一进程内所有用 stdlib ssl 的代码。这是深度学习框架单方面施加的全局降级 | 删除；证书问题用 SSL_CERT_FILE/certifi 或只对自己的 opener 生效 | 关键 |
| 归档解压无路径过滤 | extractall 共 16 处：`misc.py:140/143/147/154`、`compile_extern.py:88/92/184/449/539/609`、`install_cuda.py:193/197`、`install_msvc.py:16`、`rocm_compiler.py:49/52`、`dataset/mnist.py:195` | 归档内 `../` 可写出目录之外；Python 3.14 起默认 filter 变更行为还会再变 | 统一 `extractall(filter="data")` | 主要 |
| URL/文件名/MD5 三元组在三个文件里各写一遍 | `compile_extern.py:52/55/63/67/70/174/429/519/589`、`install_cuda.py:144-169`、`install_msvc.py:11`、`tools/release/pack_offline.py:17-46` | 必然漂移。实测 pack_offline.py 的 URL 列表没有 cutlass.zip 也没有 jtcuda 与 msvc.zip——所谓离线包在任何 CUDA 机器上仍会联网 | 一张 manifest（url、filename、sha256、平台），三方共用 | 主要 |
| 下载一个没有任何代码使用的 cutlass | `compile_extern.py:977` 无条件调 setup_cutlass，编译步骤整段被注释（`:545-560`）；仓库中 `extern/cuda/cutlass` 目录不存在，全仓无 cutlass 引用 | 每台新 CUDA 机器与每次 CUDA 门禁白下载并解压几十 MB 源码包 | 删除 setup_cutlass 与 use_cutlass | 主要 |
| 校验用 MD5，且 jittor_utils.download 完全不校验 | `misc.py:88-99` 全部 md5；`jittor_utils/__init__.py:420-426` 的 download 无哈希且以"文件大于 100 字节即视为完成"判定（当前是死代码） | MD5 抗碰撞已破，配合 TLS 降级威胁模型进一步恶化；死代码里的坏模式会被复制 | 换 SHA-256；删掉未使用的 download | 主要 |
| 下载即使注定不用也要先下 | `compile_extern.py:597-605` 先下 NCCL，之后才检查设备数与是否在 MPI 内 | 无 GPU 或非 MPI 环境白下载 | 条件前移 | 次要 |
| MKL 安装期编译并运行一个上游示例程序 | `compile_extern.py:106-107` 在 examples 目录 g++ 编译并运行，失败即 `assert 0 == os.system(...)` | 在 import 路径上编译并执行第三方二进制；诊断只有一个 AssertionError | 换成 dlopen 加符号存在性检查 | 次要 |

## 环境变量作为配置
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 76 个 C++ flag 全部可由同名**小写**环境变量设置 | `src/utils/log.h:226` `init_##name(get_from_env<type>(#name, default))`；名单含 name、no_grad、sync_run、device_id、log_file、th_mode、cache_path、python_path、cc_flags 等 | 无命名空间。shell 里恰好导出了 `name` 或 `debug` 就会改变框架行为，且只在 log_v>0 时才打印；解析失败只 LOGw 后用默认值 | 统一 JT_ 前缀（旧名保留并告警）；启动时把非默认 flag 打成一行摘要 | 主要 |
| 同名变量在 Python 与 C++ 里语义不同 | `compiler.py:1116-1117` 环境里的 cc_flags 被**追加**；同名 C++ flag 则是**整体替换**（`log.h:226`），随后又被 `compiler.py:1490` 覆写。nvcc_flags 同理 | 用户设 `cc_flags=-O1` 得到追加，设 cache_path 得到短暂替换后被覆盖，无一处文档说明 | 构建期变量与运行期 flag 分成两个命名空间 | 主要 |
| 大小写与前缀四套并存 | 小写：cc_path、nvcc_path、cache_name、debug、enable_lto、kernel_flags、use_mkl、conv_opt、log_v；大写无前缀：CUTT_PATH、CUTLASS_PATH、DISABLE_MULTIPROCESSING、FIX_TORCH_ERROR；JITTOR_ 前缀；JT_ 前缀 | 没有任何一处能查全；`debug=1` 这种名字在任何 CI 里都可能被误设 | 同上，并生成自动导出的变量清单 | 主要 |
| 导入过程反向写环境变量 | `jittor_utils/__init__.py:519` 写 cache_name、`:758` 写 cc_path；`install_cuda.py:178-179` 无条件追加 LD_LIBRARY_PATH（重复 exec 会不断变长）、`:177` 把 lib64 塞进 sys.path；`compile_extern.py:655` 写 NCCL_P2P_DISABLE、`:931` 写 use_mpi | 污染被继承到用户子进程（含 DataLoader worker、torchrun），行为随父进程是否 import 过 jittor 变化 | 需要传给子进程的配置显式构造 env 传参 | 主要 |
| 一个坏掉且无人读的环境变量 | `compiler.py:1057-1058` `os.environ["cuda_arch"] = " ".join(cu)`，cu 是字符串，结果是 `'c u 1 2 . 2 _ s m _ 8 0'`；全仓无读取方 | 死代码且往每个子进程注入垃圾 | 删除 | 次要 |
| flag 拼装顺序是巧合 | `compiler.py:1131` `kernel_opt_flags = env("kernel_flags") + opt_flags`，此时 opt_flags 还是空串（`:1121` 定义，`:1229-1233` 才填充） | kernel flags 拿不到 -O2，靠 `:1234` 单独追加的 -Ofast 兜底；用户设了含 -O 的 cc_flags 时两者都不加 | flag 组装收进一个函数一次性求值 | 次要 |

## 安装、打包与版本兼容
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 硬拒 cuDNN 9 | `compile_extern.py:337-341` major>=9 直接 raise | 错误信息建议装 `jittor[cuda12]`，而该 extra 钉死 `nvidia-cudnn-cu12==8.9.7.29`（`pyproject.toml:37`、`cuda_wheel.py:37`），**与任何现代 torch 的依赖直接冲突**，而 shim 的定位恰恰是与 torch 生态共存 | 迁 RNN 到 v8 API 后放开；在此之前错误信息要说明与 torch 共存的实际后果 | 主要 |
| CUDA wheel 栈任何一项不匹配都静默降级 | `cuda_wheel.py:270-273` `except (CudaWheelError, PackageNotFoundError): return None`（除非 STRICT）；已构造好的诊断字符串（`:241-244,256-259`）被丢弃 | 用户装了 `jittor[cuda12]` 后仍走系统 CUDA 然后撞上 cuDNN 9 报错，两件事看不出因果 | 失败时至少 LOG.w 出原因；strict 应为默认 | 主要 |
| 新架 GPU 不生成 PTX 回退 | `compiler.py:1466-1472` 硬编码 `max_arch = 90`（超出只 warn 并钳到 90）；`:1485-1486` 只发 `-arch=compute_{min} -code=sm_{x}`，产物无 PTX | Blackwell（sm_100/120）上没有可用 cubin 也没有可 JIT 的 PTX，报 no kernel image；`install_cuda.py:155` 的版本表停在 CUDA 12.2 | 用 `-gencode arch=compute_X,code=[sm_X,compute_X]` 保留 PTX；max_arch 由 nvcc 查询 | 主要 |
| 声称的版本与仓库分支不一致 | `python/jittor/__init__.py:12` `__version__ = '1.3.11.0'`；缓存键取 `rsplit('.',1)[0]` = jt1.3.11 | 分支叫 2.0 而产物版本是 1.3.11.0；缓存不按 patch 版本分区 | 明确 2.0 的版本策略 | 主要 |
| 发布流水线从不真正跑一次 jittor | `.github/workflows/release.yml:145-190` 安装 wheel 后只校验版本号与三个资源文件存在；selftest 只在 nox packaging（`noxfile.py:805`）里跑，不在 release 流程内 | 可以发布一个根本编译不起来的 wheel | release 的 platform-validation 阶段加一次 selftest | 主要 |
| 仓库内提交预编译目标文件 | `extern/rocm/rocm_cache.tar.gz`（115 KB，含两个 .o），由 `rocm_compiler.py:44-52` 解压后直接链接进核心，且随 MANIFEST.in 的 recursive-include 进入每个 wheel | 无来源无构建脚本的二进制进入所有用户的 wheel 并被链接进进程 | 从源码构建，否则至少给构建脚本与来源说明 | 主要 |
| README 没有说明首次导入的真实代价 | `README.md:97-130` 只有 pip install 加 selftest | 未提及需联网、需 1–2 GB 缓存、首次 import 可能十几分钟、可能自动下载 CUDA、git 是否存在会改变缓存路径 | 安装章节加"首次运行会发生什么"与离线安装说明 | 主要 |

## 跨平台与死代码
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| Windows + MinGW 分支引用未定义变量 | `compiler.py:83` `link = link + ...`，函数内外都没有 link 的定义 | 该分支一执行必 NameError，说明从未被跑过 | 删掉或补测试 | 次要 |
| cuda_wheel 里的 Darwin 分支不可达 | `cuda_wheel.py:239` 非 posix 直接 return None，但 find_library（`:154-159`）仍分 nt / Darwin 两支；NVIDIA 不发布 macOS CUDA wheel | 维护成本，读者误以为支持 | 删除 | 次要 |
| corex/rocm 后端每次 import 都被加载并探测 | `compiler.py:1324-1333` 无条件 import 三个后端并逐个 check()；`corex_compiler.py:86` 只查目录是否存在；`rocm_compiler.py:18` 模块级找 hipcc | 所有用户为两个几乎无人使用的后端付探测成本；后端注册是硬编码而非入口点 | 改 entry_point 发现加懒加载 | 次要 |
| env_or_try_find 在两处重复定义 | `jittor_utils/__init__.py:573` 与 `compiler.py:949` 函数体完全一致，后者遮蔽前者 | 修一处漏一处 | 删掉副本 | 次要 |
| 每个 CPU kernel 编译都要起一个 Python 进程 | `src/jit_compiler.cc:253-256` Linux 上 CPU kernel 命令被包成 `python asm_tuner.py`；`asm_tuner.py:145-160` 先编成 .post.s（带 -g）、Python 文本改写、再汇编成 .so | 冷启动 kernel 编译成本 2–3 倍；整套机制只服务一个用途：`use_movnt_pass.cc:24` 那一条正则 | movnt 改写做成 intrinsic 或编译器选项，删掉 asm_tuner 链路 | 主要 |

## 门禁与开发工具链
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 每个 session 都从空缓存起步且必须联网 | `noxfile.py:346-363` `_session_env` 把 HOME/JITTOR_HOME/XDG_CACHE_HOME/TMPDIR 全指向新建的临时目录并先 rmtree | 每次门禁重编整个核心并重新从清华镜像下载 mkl（CPU）与 cutt/cutlass（CUDA）。这是 40 分钟与 2 小时的主要构成，同时把一台中国高校主机变成 CI 的硬依赖 | 缓存目录改成按构建配置指纹命名的共享目录；第三方包本地镜像预置 | 主要 |
| 每个测试目标一个 pytest 进程，串行执行 | `noxfile.py:410-422` 对 defaults 逐项 session.run；CPU 18 项加 oracle 9 项共 27 次进程启动，每次都要 import jittor；`stop_on_first_error = True`（`:294`） | 进程启动开销乘以目标数；无 xdist；第一个失败即停，一轮只能拿到一个失败信息 | 同模式目标合并成一次调用加 xdist；分 smoke/full 两层 | 主要 |
| 门禁是手工维护的测试白名单 | `noxfile.py:152-289` 逐条列出路径。统计：tests/ 下 289 个 test_*.py，只有 86 个能被任一 session 默认目标触及，**203 个从不在门禁里跑** | 新增测试默认是死的；专门守护冷启动双映射这个已修 bug 的 `tests/compiler/test_cold_start_runtime.py` 就不在门禁内 | 默认跑整个 tests/，用 marker 做减法而不是白名单做加法 | 主要 |
| 仓库检查脚本是只增不减的历史文件名黑名单 | `agent/scripts/check_repo_layout.sh` 40+ 条 forbidden legacy path（`:90-124`）、根目录白名单（`:72-88`）、8 组全树 grep（`:225-249`），并用字符串拼接避免匹配到脚本自身 | 每次重构只增不减；新增任何根目录文件都会失败；迁移文档提到历史路径会误报；全树 grep 使这个"快速"门禁并不快 | 用 git 历史记录已删除路径，只保留少数真会复发的检查 | 主要 |
| pytest 配置强制注入主源码树 | `pyproject.toml` `pythonpath = ["python"]` | 副本或 worktree 里跑 pytest 会导入主树 | conftest 按环境变量决定 | 次要 |
| 门禁一律关掉并行编译器 | `noxfile.py:687/1128/1548`、`tools/run_test_suite.py:59` | 门禁验证的不是用户默认跑到的代码路径（默认值 16） | 修好同步原语后默认打开，另设串行 session 做对照 | 主要 |

## 新用户从 pip install 到跑通第一次训练：至少 17 个失败点，只有 4 个信息可操作
| # | 失败点 | 失败信息 | 可操作 |
| --- | --- | --- | --- |
| 1 | 没有 g++ | `RuntimeError: g++ not found`（`jittor_utils/__init__.py:554`） | 是 |
| 2 | 没有 python3-dev | 明确列出搜索路径并提示装 python3.x-dev（`:653-656`） | **是，写得很好** |
| 3 | 没有 OpenMP 运行库 | `assert libname is not None, "openmp library not found"`（`compiler.py:1395`） | 部分 |
| 4 | git 不在 PATH 或 detached HEAD | 无任何信息，静默改变缓存路径 | 否 |
| 5 | 有驱动无 nvcc 触发自动下载 2GB | 只有一行 Downloading，无大小无耗时无取消方式 | 否 |
| 6 | 下载失败或镜像不可达 | `Download File failed, url: ...`（`misc.py:70-74`） | 部分 |
| 7 | MD5 不匹配（代理返回错误页） | `MD5 mismatch...`（`misc.py:76`），不删坏文件不提示重试 | 否，下次运行因文件已存在反复失败 |
| 8 | 磁盘满或缓存被截断 | 无检查，表现为散布编译失败与段错误 | 否 |
| 9 | 核心编译失败 | `system_with_check` 抛完整命令行加 "This might be an overcommit issue"（`log.cc:679-682`），这条建议只对退出码异常成立，普通编译错误也带上，误导 | 部分 |
| 10 | jit_utils 需要重编 | LOG.e 后 **exit 0** | 否，脚本看起来成功了 |
| 11 | 旧的 CPU-only jittor_core 遮蔽 CUDA 版本 | 明确指出被导入的文件路径与删除方法（`compiler.py:1454-1460`） | **是，写得很好** |
| 12 | cuDNN 是 9.x | 建议装 jittor[cuda12]（`compile_extern.py:338-341`） | 部分，该建议在装了 torch 的环境会导致依赖冲突 |
| 13 | jittor[cuda12] 组件版本不完全匹配 | 静默返回 None | 否 |
| 14 | cublas/curand/cufft/cusparse 任一找不到 | LOG.f 中止；search_file 只打印 file X not found in [dirs] | 部分 |
| 15 | GPU 架构新于 nvcc 支持范围 | warn 说"will be backward-compatible"——这句话是**错的**，没有 PTX 就没有向后兼容 | 否，警告掩盖了随后的运行期失败 |
| 16 | 另一个进程持锁或孤儿锁 | 无任何输出，无限等待（F_SETLKW 无超时） | 否，实测两次各 40 分钟 |
| 17 | 单个算子 JIT 编译失败 | LOGf 带 jit_key、生成源文件路径与编译器输出（`parallel_compiler.cc:204-211`） | **是，写得很好** |

可诊断性呈两极：**已被人踩过并专门写过注释的失败点信息质量很高，其余全靠猜**。缺的不是
文案，而是一层统一的"构建前置条件检查"——在任何编译开始前一次性校验编译器、Python 头
文件、OpenMP、磁盘空间、网络可达性与 CUDA 组件版本，一次报告全部缺失项，并区分可自动
修复与需用户操作。现在这些检查散在 1500 行的模块顶层脚本里，按执行顺序一个个把用户拦下来。
