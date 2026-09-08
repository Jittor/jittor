# oneDNN v3 CPU runtime

状态：功能迁移完成，性能验收后置。oneDNN v3 的 provider、运行时 owner、
卷积前反向 primitive 与 matmul/batched matmul 路径已接入；执行前重新绑定
外部 tensor data handle，避免缓存地址。显式 `JT_BUILD_MKL_INCLUDE_PATH` /
`JT_BUILD_MKL_LIB_PATH` 支持已有安装，缺失头文件或库会失败并给出路径。

本地使用官方 oneDNN 3.9.1 源码构建安装进行 CPU 验证；不声称 NPU/CUDA 验证。
验证包含 v3 header TU、卷积前反向、矩阵乘、能力声明、selftest 和 provider
失败路径。primitive 缓存与每次重建的开销比较属于性能工作，按计划后置。
