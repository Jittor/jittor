# Jittor: a Just-in-time(JIT) deep learning framework
# Jittor: 即时编译深度学习框架

![Jittor Logo](https://cg.cs.tsinghua.edu.cn/jittor/favicon_package_v0/JittorLogo_Final1220.svg)

[Quickstart](#quickstart) | [Install](#install) | [Tutorial](#tutorial) | [Chinese](./README.cn.md)

[快速开始](#快速开始) | [安装](#安装) | [教程](#教程) | [English](./README.md)

Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model. Jittor also contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc. .

Jittor 是一个基于即时编译和元算子的高性能深度学习框架，整个框架在即时编译的同时，还集成了强大的Op编译器和调优器，为您的模型生成定制化的高性能代码。Jittor还包含了丰富的高性能模型库，涵盖范围包括：图像识别，检测，分割，生成，可微渲染，几何学习，强化学习等等。

The front-end language is Python. Module Design and Dynamic Graph Execution is used in the front-end, which is the most popular design for deeplearning framework interface. The back-end is implemented by high performance language, such as CUDA,C++.

Jittor前端语言为Python。前端使用了模块化和动态图执行的设计，这是目前最主流的深度学习框架接口设计。后端则使用高性能语言编写，如CUDA，C++。

Related Links:
*  [Jittor Website](https://cg.cs.tsinghua.edu.cn/jittor/)
*  [Jittor Tutorials](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/)
*  [Jittor Models](https://cg.cs.tsinghua.edu.cn/jittor/resources/)
*  [Jittor Documents](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)
*  [Github](https://github.com/jittor/jittor), [GitLink](https://www.gitlink.org.cn/jittor/jittor), [Gitee](https://gitee.com/jittor/jittor)
*  [Jittor Forum](https://discuss.jittor.org/)
*  [Awesome Jittor List](https://github.com/Jittor/jittor/blob/master/AWESOME-JITTOR-LIST.md)
*  IM: QQ Group(761222083)

相关链接：
*  [Jittor官网](https://cg.cs.tsinghua.edu.cn/jittor/)
*  [Jittor教程](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/)
*  [Jittor模型库](https://cg.cs.tsinghua.edu.cn/jittor/resources/)
*  [Jittor文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)
*  [Github](https://github.com/jittor/jittor)， [GitLink](https://www.gitlink.org.cn/jittor/jittor)， [Gitee](https://gitee.com/jittor/jittor)
*  [Jittor 论坛](https://discuss.jittor.org/)
*  [Jittor 精选仓库](https://github.com/Jittor/jittor/blob/master/AWESOME-JITTOR-LIST.md)
*  即时通信: QQ Group(761222083)


The following example shows how to model a two-layer neural network step by step and train from scratch In a few lines of Python code.

下面的代码演示了如何一步一步使用Python代码，从头对一个双层神经网络建模。

```python
import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np

class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 10)
        self.relu = nn.Relu() 
        self.layer2 = nn.Linear(10, 1)
    def execute (self,x) :
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def get_data(n): # generate random data for training test.
    for i in range(n):
        x = np.random.rand(batch_size, 1)
        y = x*x
        yield jt.float32(x), jt.float32(y)


learning_rate = 0.1
batch_size = 50
n = 1000

model = Model()
optim = nn.SGD(model.parameters(), learning_rate)

for i,(x,y) in enumerate(get_data(n)):
    pred_y = model(x)
    dy = pred_y - y
    loss = dy * dy
    loss_mean = loss.mean()
    optim.step(loss_mean)
    print(f"step {i}, loss = {loss_mean.data.sum()}")
```

## Contents

* [Quickstart](#quickstart)
* [Install](#install)
* [Tutorial](#tutorial)
* [Contributing](#contributing)
* [The Team](#theteam)
* [License](#license)

## 大纲

- [快速开始](#快速开始)
- [安装](#安装)
- [教程](#教程)
- [贡献](#贡献)
- [团队](#团队)
- [版权声明](#版权声明)

## Quickstart

## 快速开始

We provide some jupyter notebooks to help you quick start with Jittor.

我们提供了一些jupyter notebooks来帮助您快速入门Jittor。

- [Example: Model definition and training][1]
- [示例：模型定义与训练][1]
- [Basics: Op, Var][2]
- [基础：Op, Var][2]
- [Meta-operator: Implement your own convolution with Meta-operator][3]
- [元算子：通过元算子实现自己的卷积层][3]

## Install

## 安装

Jittor框架对环境要求如下:

Jittor environment requirements:

| OS                                                     | CPU                                 | Python | Compiler     | (Optional) GPU platform                                |
|--------------------------------------------------------|-------------------------------------|--------|--------------|---------------------------------------------|
| Linux<br>(Ubuntu, CentOS, Arch, <br>UOS, KylinOS, ...) | x86 <br>x86_64 <br>ARM <br>loongson | >= 3.7 | g++ >=5.4    | Nvidia CUDA >= 10.0, [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar) <br> or [AMD ROCm](https://docs.amd.com/) >= 4.0 <br> or [Hygon DCU DTK](https://tycloud.hpccube.com/doc/1.0.6/11277/general-handbook/software-tutorial/jittor.html) >= 22.04 |
| macOS <br>(>= 10.14 Mojave)                            | intel<br>Apple Silicon              | >= 3.7 | clang >= 8.0 | -                                           |
| Windows 10 & 11                                        | x86_64                              | [>= 3.8](https://www.python.org/downloads/windows/) | -            | Nvidia CUDA >= 10.2 [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)                               |

Jittor 提供了三种安装方法：pip、docker和手动安装：

Jittor offers three ways to install: pip, docker, or manual.

## Pip 安装

## Pip install

下面将展示Ubuntu的安装命令，如果您在使用其他Linux操作系统（如CentOS）， 请安装好依赖（Python>=3.7, g++>=5.4）或者使用**docker安装**， 如果您已经装好编译器和对应版本的Python,我们强烈推荐您使用这种方法
(如果无法访问github, 可以通过Jittor主页下载):

```bash
sudo apt install python3.7-dev libomp-dev
python3.7 -m pip install jittor
# or install from github(latest version)
# python3.7 -m pip install git+https://github.com/Jittor/jittor.git
python3.7 -m jittor.test.test_example
```

如果测试运行通过,恭喜你已经安装完成.
jittor会自动在路径中寻找合适的编译器, 如果您希望手动指定编译器, 请使用环境变量 `cc_path` 和 `nvcc_path`(可选).

### macOS 安装

### macOS install

macOS 请使用 [homebrew](https://brew.sh) 安装额外的依赖。

Please first install additional dependencies with [homebrew](https://brew.sh).

```bash
brew install libomp
```

之后您可以通过 pip 安装 jittor，并测试是否可以成功运行。

Then you can install jittor through pip and run the example.

```bash
python3.7 -m pip install jittor
python3.7 -m jittor.test.test_example
```

目前在 macOS 中，jittor 只支持 CPU 计算。

Currently jittor only supports CPU on macOS.

### Windows安装

### Windows install

Windows 请准备好Python>=3.8，安装方法如下(conda安装需要额外命令)：

Windows user please prepare Python>=3.8, install instructions are list below(conda needs extra instructions)：

```bash
# check your python version(>=3.8)
python --version
python -m pip install jittor
# if conda is used
conda install pywin32
```

Windows 下，jittor会自动检测显卡并安装对应的 CUDA， 请确保您的NVIDIA驱动支持CUDA 10.2 以上，您还可以使用如下命令手动为Jittor安装CUDA：

In Windows, jittor will automatically detect and install CUDA, please make sure your NVIDIA driver support CUDA 10.2  or above, or you can manually let jittor install CUDA for you:

```bash
python -m jittor_utils.install_cuda
```


## Docker Install

## Docker 安装

我们提供了Docker安装方式，免去您配置环境，Docker安装方法如下：

We provide a Docker installation method to save you from configuring the environment. The Docker installation method is as follows:

```
# CPU only(Linux)
docker run -it --network host jittor/jittor
# CPU and CUDA(Linux)
docker run -it --network host --gpus all jittor/jittor-cuda
# CPU only(Mac and Windows)
docker run -it -p 8888:8888 jittor/jittor
```

关于Docker安装的详细教程，可以参考[Windows/Mac/Linux通过Docker安装计图](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-15-00-00-docker/)

## 手动安装
## manual install

We will show how to install Jittor in Ubuntu 16.04 step by step, Other Linux distributions may have similar commands.

我们将逐步演示如何在Ubuntu 16.04中安装Jittor，其他Linux发行版可能可以使用类似的命令。

### Step 1: Choose your back-end compiler

### 步骤一：选择您的后端编译器

```bash
# g++
sudo apt install g++ build-essential libomp-dev

# OR clang++-8
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install_llvm.sh > /tmp/llvm.sh
bash /tmp/llvm.sh 8
```
### Step 2: Install Python and python-dev

### 步骤二：安装Python和python-dev

Jittor need python version >= 3.7.

Jittor需要python的版本>=3.7。

```bash
sudo apt install python3.7 python3.7-dev
```

### Step 3: Run Jittor

### 步骤三：运行Jittor

The whole framework is compiled Just-in-time. Let's install jittor via pip

整个框架是即时编译的。 让我们通过pip安装jittor

```bash
git clone https://github.com/Jittor/jittor.git
sudo pip3.7 install ./jittor
export cc_path="clang++-8"
# if other compiler is used, change cc_path
# export cc_path="g++"
# export cc_path="icc"

# run a simple test
python3.7 -m jittor.test.test_example
```
if the test is passed, your Jittor is ready.

如果通过了测试，那么您的Jittor已经准备就绪。

### Optional Step 4: Enable CUDA

### 可选步骤四：启用CUDA

Using CUDA in Jittor is very simple, Just setup environment value `nvcc_path`

在Jittor中使用CUDA非常简单，只需设置环境值`nvcc_path`

```bash
# replace this var with your nvcc location 
export nvcc_path="/usr/local/cuda/bin/nvcc" 
# run a simple cuda test
python3.7 -m jittor.test.test_cuda 
```
if the test is passed, your can use Jittor with CUDA by setting `use_cuda` flag.

如果测试通过，则可以通过设置`use_cuda`标识符在Jittor中启用CUDA。

```python
import jittor as jt
jt.flags.use_cuda = 1
```

### Optional Step 5: Test Resnet18 training

### 可选步骤五：测试训练Resnet18

To check the integrity of Jittor, you can run Resnet18 training test. Note: 6G GPU RAM is requires in this test.

要检查Jittor的完整性，您可以运行Resnet18训练测试。需要注意的是，这个测试需要6G显存。

```bash
python3.7 -m jittor.test.test_resnet
```
if those tests are failed, please report bugs for us, and feel free to contribute ^_^

如果这些测试失败，请为我们报告错误，我们十分欢迎您为Jittor做出贡献^ _ ^

## Tutorial

## 教程

In the tutorial section, we will briefly explain the basic concept of Jittor.

在教程部分，我们将简要解释Jittor的基本概念。

To train your model with Jittor, there are only three main concepts you need to know:

要使用Jittor训练模型，您需要了解两个主要概念：

* Var: basic data type of jittor
* Var：Jittor的基本数据类型
* Operations: Jittor'op is simular with numpy
* Operations：Jittor的算子与numpy类似

### Var

### 数据类型

First, let's get started with Var. Var is the basic data type of jittor. Computation process in Jittor is asynchronous for optimization. If you want to access the data, `Var.data` can be used for synchronous data accessing.

首先，让我们开始使用Var。Var是jittor的基本数据类型，为了运算更加高效Jittor中的计算过程是异步的。 如果要访问数据，可以使用`Var.data`进行同步数据访问。

```python
import jittor as jt
a = jt.float32([1,2,3])
print (a)
print (a.data)
# Output: float32[3,]
# Output: [ 1. 2. 3.]
```

And we can give the variable a name.

此外我们可以给变量起一个名字。

```python
a.name('a')
print(a.name())
# Output: a
```

### Operations

### 数据运算

Jittor'op is simular with numpy. Let's try some operations. We create Var `a` and `b` via operation `jt.float32`, and add them. Printing those variables shows they have the same shape and dtype.

 Jittor的算子与numpy类似。 让我们尝试一些运算， 我们通过Op`jt.float32`创建Var `a`和`b`，并将它们相加。 输出这些变量相关信息，可以看出它们具有相同的形状和类型。

```python
import jittor as jt
a = jt.float32([1,2,3])
b = jt.float32([4,5,6])
c = a*b
print(a,b,c)
print(type(a), type(b), type(c))
# Output: float32[3,] float32[3,] float32[3,]
# Output: <class 'jittor_core.Var'> <class 'jittor_core.Var'> <class 'jittor_core.Var'>
```
Beside that, All the operators we used `jt.xxx(Var, ...)` have alias `Var.xxx(...)`. For example:

除此之外，我们使用的所有算子`jt.xxx(Var,...)`都具有别名`Var.xxx(...)`。 例如：

```python
c.max() # alias of jt.max(c)
c.add(a) # alias of jt.add(c, a)
c.min(keepdims=True) # alias of jt.min(c, keepdims=True)
```

if you want to know all the operation which Jittor supports. try `help(jt.ops)`. All the operation you found in `jt.ops.xxx`, can be used via alias `jt.xxx`.

如果您想知道Jittor支持的所有运算，可以运行`help(jt.ops)`。 您在`jt.ops.xxx`中找到的所有运算都可以通过别名`jt.xxx`。

```python
help(jt.ops)
# Output:
#   abs(x: core.Var) -> core.Var
#   add(x: core.Var, y: core.Var) -> core.Var
#   array(data: array) -> core.Var
#   binary(x: core.Var, y: core.Var, op: str) -> core.Var
#   ......
```
### More

### 更多教程

If you want to know more about Jittor, please check out the notebooks below:

如果您想进一步了解Jittor，请查看以下notebooks：

* Quickstart
    - [Example: Model definition and training][1]
    - [Basics: Op, Var][2]
    - [Meta-operator: Implement your own convolution with Meta-operator][3]
* 快速开始
    * [示例：模型定义与训练][1]
    * [基本概念：Op, Var][2]
    * [元算子：通过元算子实现自己的卷积层][3]
* Advanced
    - [Custom Op: write your operator with C++ and CUDA and JIT compile it][4]
    - [Profiler: Profiling your model][5]
    - Jtune: Tool for performance tuning
* 进阶
    * [自定义算子：使用C ++和CUDA编写您的算子，并其进行即时编译][4]
    * [性能分析器：分析您的模型][5]
    * Jtune：性能调优工具



[1]: python/jittor/notebook/example.src.md	"example"
[2]: python/jittor/notebook/basics.src.md	"basics"
[3]: python/jittor/notebook/meta_op.src.md	"meta_op"
[4]: python/jittor/notebook/custom_op.src.md	"custom_op"
[5]: python/jittor/notebook/profiler.src.md	"profiler"
[1]: python/jittor/notebook/example.src.md	"示例"
[2]: python/jittor/notebook/basics.src.md	"基本概念"
[3]: python/jittor/notebook/meta_op.src.md	"元算子"
[4]: python/jittor/notebook/custom_op.src.md	"自定义算子"
[5]: python/jittor/notebook/profiler.src.md	"性能分析器"

Those notebooks can be started in your own computer by `python3.7 -m jittor.notebook`

这些notebooks可以通过python3.7 -m jittor.notebook在您自己的计算机中运行。

## Contributing

## 贡献

Jittor is still young. It may contain bugs and issues. Please report them in our bug track system. Contributions are welcome. Besides, if you have any ideas about Jittor, please let us know.

Jittor还很年轻。 它可能存在错误和问题。 请在我们的错误跟踪系统中报告它们。 我们欢迎您为Jittor做出贡献。 此外，如果您对Jittor有任何想法，请告诉我们。

您可以用以下方式帮助Jittor：

* 在论文中引用 Jittor
* 向身边的好朋友推荐 Jittor
* 贡献代码
* 贡献教程和文档
* 提出issue
* 回答 jittor 相关问题
* 点亮小星星
* 持续关注 jittor
* ……

You can help Jittor in the following ways:

* Citing Jittor in your paper
* recommend Jittor to your friends
* Contributing code
* Contributed tutorials and documentation
* File an issue
* Answer jittor related questions
* Light up the stars
* Keep an eye on jittor
* ......

## Contact Us

## 联系我们

官方主页： http://cg.cs.tsinghua.edu.cn/jittor/

电子邮件：jittor@qq.com

提出issue：https://github.com/Jittor/jittor/issues

Website: http://cg.cs.tsinghua.edu.cn/jittor/

Email: jittor@qq.com

File an issue: https://github.com/Jittor/jittor/issues

QQ Group: 761222083

QQ 群：761222083

<img src="https://cg.cs.tsinghua.edu.cn/jittor/images/news/2020-12-8-21-19-1_2_2/fig4.png" width="200"/>

## The Team

## 团队

Jittor is currently maintained by the [Tsinghua CSCG Group](https://cg.cs.tsinghua.edu.cn/). If you are also interested in Jittor and want to improve it, Please join us!

Jittor目前由[清华大学计算机图形学组](https://cg.cs.tsinghua.edu.cn/)维护。 如果您也对Jittor感兴趣并希望对其进行改进，请加入我们！

## Citation

## 引用

```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```

## License

## 版权声明

Jittor is Apache 2.0 licensed, as found in the LICENSE.txt file.

如LICENSE.txt文件中所示，Jittor使用Apache 2.0版权协议。
