# JIT 目标与转置视图整项收口

- 状态：3.04、5.03原验收完成；不等同于5.02任意stride模型完成。
- 日期：2026-09-08；owner：协调分区；基线：69c533404。
- 复查：执行计划、key preparation或VarView变更时。

3.04：ExecPlan保存BackendId，可恢复的ExecutionBackendScope把目标传到编译worker、
CodeOp、CopyOp和codegen pass/tuner。prepare不再清CPU/CUDA能力位；普通Op不再发布
编译前后的双key，编译前后检查key恒定。Fused合法调优的映射保留。ACL CPU fallback
也改为显式scope，保留runtime/context/tuned回滚；未做ACL硬件验证。

5.03：VarView记录Slice/Transpose步骤，源assign/update刷新活转置视图，写回应用逆
axes。matmul从真实view关系取源与last-two-axis信息，删除三个旧隐藏标记。保留七个
元算子和统一图，不为转置另造计算图。原strict xfail改为正常回归。

一次统一CPU编译与定向13 passed（39.49秒）：key purity1、CodeOp来源/前反向4、
fused2、transpose6。GPU6/7同批7 passed（77.36秒）：transpose6和key purity1；
CUDA purity让同一双源码Op先经CPU prepare再实际执行CUDA，分别以17/29区别两个实现。
转置覆盖源替换、已物化后反复赋值、transpose+slice双向写回、batched融合及一般置换梯度。
24个关键/相关codegen TU host语法通过，ACL scope异常恢复抽取合同1 passed。
首编两处NanoVector不可赋索引已修后在原缓存增量通过，未重复冷建。

CPU日志/JUnit（未版本化）：`/home/zy/jittor-lab/_state/purity-cpu-qtgiul/`。
CUDA结果来自工具会话75019，未另存日志/JUnit，不为补日志重跑；复查节点为
`test_transpose_view_staleness.py`与`test_jit_prepare_purity.py`的CUDA参数项。
未执行全量suite或新wheel验收。

另外回填1.05/2.23/3.24已完成状态：旧看板仍停在preflight阻塞，后续4.15和独立core
wheel的207 TU冷构建早已解除。当前实际目录逐条符合原布局要求。这三项是历史状态
纠正，不计为本批新增实现，证据见[独立发行物记录](2026-09-08-independent-compat-distribution.md)。
