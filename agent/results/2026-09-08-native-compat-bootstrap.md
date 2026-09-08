# Native 启动与可选 compat 解耦

7.12 的物理拆包前置已落地，整项仍未完成。

原生别名表和共享 loader 由 `_runtime/import_aliases.py` 持有；compat 只登记
自己的别名和激活回调。原生 build 不再反向依赖 compat。启动经 stdlib-only
`compat_bootstrap` 桥在显式请求时加载兼容包；原生导入不加载 Triton 兼容域。
历史 Torch 别名保留按需加载和对象身份。请求缺失的兼容包时给出安装错误。

轻量验证：CPU 11 passed（4.18 s），CUDA 6 passed（18.64 s）。包括阻止全部
`jittor.compat` 导入后 native 前向/反向仍可运行、缺包错误、旧别名按需加载和
独立 Torch 入口。未运行完整套件，也未构建本批 wheel。

日志目录：`/home/zy/jittor-lab/_state/independent-deploy-v1SgCm/`，文件为
`native-bootstrap-final-cpu.log` 和 `native-bootstrap-cuda.log`。

下一步将 `python/jittor/compat` 整树迁到顶层独立 distribution，分开 core 与
compat 的包文件、资源和部署入口，并用隔离两包安装检查导入。对象状态归并与
native Torch 角色依赖退出仍是独立的未完成要求。
