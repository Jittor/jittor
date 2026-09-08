# Transformers / ms-swift / verl 生态验证

状态：环境验证与阻塞定位，不能关闭生态任务。

`transformers==4.56.2` 在开发环境可导入；`ms-swift` 的 `swift==4.5.2` 只在
Jittor Python 3.11 环境存在，独立 PyTorch oracle 环境没有该包；`verl` 两侧均未安装。
因此 ms-swift 与 verl 尚无独立 Torch 前反向对拍证据。

GPT-2 使用独立 PyTorch 2.12.1+cu126 oracle，在临时隔离不匹配的 torchvision 后可构建。
Jittor shim forward 可运行，但 backward 在复杂 residual/attention/tied lm_head 图触发
`backward liveness release without a matching owner`。单层 Linear/平方/求和 backward
正常，说明问题位于复杂图的 holder/liveness ownership；该缺陷正在单独定位，不能通过
关闭检查或改变测试期望掩盖。

生态测试已修复 oracle 环境隔离与可选 torchvision 处理，避免把 Jittor facade 当成 PyTorch
参考实现。当前没有速度结论；vLLM/TRELLIS/verl/ms-swift 需在安装匹配依赖的环境重新验证。
