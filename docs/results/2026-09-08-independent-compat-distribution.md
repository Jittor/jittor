# compat 独立发行物

- 状态：物理拆包完成；7.12 对象状态与native角色依赖仍未完成，7.18 的vLLM外部仓库提取仍未完成。
- 日期：2026-09-08；owner：协调分区；基线：9689b14d5。
- 复查触发：包布局、资源或激活入口变化。

`python/jittor/compat`整树进入顶层`compat/`，以`jittor-torch`独立构建，保留
`jittor.compat`导入名。core不再包含兼容文件/资源/部署命令。独立wheel提供`torch`，
与deploy共用`shim/resources/torch/__init__.py`，两distribution不共同拥有任何文件。
扩展头文件、C++源码与stub资源显式列入独立包，修复首次打包发现的资源遗漏。
PEP517构建明确排除运行期setup模块，避免直接wheel与sdist重建结果不同。

源码测试、lint扫描、wheel/sdist审计按两个owner更新；nox显式安装对应checkout的
compat editable，standalone runner遇到缺包/错误checkout给出安装命令。Nox源码复制
同时修复把原生`python/jittor/build`误判成生成物的问题。未重跑全套nox或全仓测试。

验证：源码启动/独立frontend/deploy共21 passed（5.54秒），packaging/import-layering
21 passed（1.89秒）。core wheel audit 1010 paths；compat 151 paths；两wheel交集0。
兼容源码141个Python/C++/头文件与wheel逐字节一致，部署入口与顶层torch入口一致。
compat sdist重建wheel与直接wheel151个成员无新增、删除或字节差异。
core-only隔离安装后冷编译207 TU，native前向/反向通过；随后在同一target一次合装
两个wheel，torch-first反向与native-first数值/原生Var身份通过。
本批不修改算子数学；未新增CUDA/NPU实机验收，未声称完整Torch兼容已完成。

产物与日志（未版本化）：`/home/zy/jittor-lab/_state/compat-dist-OIR3DI/`。

| 产物 | SHA-256 |
| --- | --- |
| source/dist/jittor-1.3.11.0-py3-none-any.whl | 3d85e887e0f2fd33728d5493a82acc31d262483a8cb07f8dbcb0b019e540bf60 |
| source/compat/dist/jittor_torch-1.3.11.0-py3-none-any.whl | 9927110d4366cc7dc43a42a193124fde66faae8b3ace360dcae4e86986d0d620 |

换机源码开发：`python -m pip install -e . -e ./compat`。仅需native则只安装core。
离线安装：将上述两wheel一起交给`pip install --no-deps`；依赖需事先按目标机器准备。
使用`--target`时两个wheel必须同一次安装，避免后装时覆盖共同的jittor目录。
完整用户入口说明见[Shim指南](../compatibility/torch-shim.md)。
