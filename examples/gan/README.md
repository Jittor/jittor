# Conditional GAN Web Example

`simple_cgan.py` 是一个自带训练的条件生成对抗网络示例：它**先在启动时训练**一个小模型，
再把模型接到命令行或 Web 界面上。不下载任何东西，也不需要显卡。

模型和任务是教程
[`examples/notebooks/conditional_gan.md`](../notebooks/conditional_gan.md) 里那一套：8×8 的
灰度图，条件 `k` 表示「第 k 列是亮带」。保留这个刻意缩小的问题，是为了让整个示例**离线
可跑、一分钟内能训练完**，从而把重点放在「训练一个条件生成器，再把它接进应用」这条链路
上。真实的图像生成请按教程第 10 节换成 MNIST。

脚本在 **import 阶段不产生任何副作用**：`jittor`、`PIL`、`pywebio` 都在 `main()` 里才导入，
所以它可以安全地被打包进 sdist、被工具探查。

## 只看结果，不开服务

```bash
python examples/gan/simple_cgan.py --digits 01234567 \
  --output "$JITTOR_LAB_ROOT/_state/examples/cgan.png" --no-server
```

它会训练（默认 400 步，CPU 上约一分钟），然后把 `--digits` 里每个条件各生成一格，横向拼
成一张图写到 `--output`。

## 起一个只在本地监听的服务

```bash
python examples/gan/simple_cgan.py --host 127.0.0.1 --port 8123
```

启动后会用 PyWebIO 打开一个页面：输入一串数字（只能含 `0`–`7`），点 Generate，返回对应的
生成结果。PyWebIO 是可选的，装在 `requirements/examples.txt` 里。

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--digits` | `01234567` | 要生成的条件序列，只能含 `0`–`7` |
| `--steps` | `400` | 训练步数；调小可以更快看到结果 |
| `--seed` | `0` | 随机种子，同一台机器上结果可复现 |
| `--host` / `--port` | `127.0.0.1` / `8123` | Web 服务的监听地址与端口 |
| `--output` | 无 | 把结果写成一个 PNG |
| `--no-server` | 关 | 训练完就退出，不起服务 |
