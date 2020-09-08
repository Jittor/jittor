# Jittor: a Just-in-time(JIT) deep learning framework
# Jittor: 即时编译深度学习框架

[Quickstart](#quickstart) | [Install](#install) | [Tutorial](#tutorial) | [Chinese](./README.cn.md)

[快速开始](#快速开始) | [安装](#安装) | [教程](#教程)

Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model.

Jittor 是一个基于即时编译和元算子的高性能深度学习框架，整个框架在即时编译的同时，还集成了强大的Op编译器和调优器，为您的模型生成定制化的高性能代码。

The front-end language is Python. Module Design is used in the front-end, which is the most popular design for deeplearning framework interface. The back-end is implemented by high performance language, such as CUDA,C++.

Jittor前端语言为Python。前端使用了模块化的设计，这是目前最主流的深度学习框架接口设计。后端则使用高性能语言编写，如CUDA，C++。

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

我们提供了一些jupyterr notebooks来帮助您快速入门Jittor。

- [Example: Model definition and training][1]
- [示例：模型定义与训练][1]
- [Basics: Op, Var][2]
- [基础：Op, Var][2]
- [Meta-operator: Implement your own convolution with Meta-operator][3]
- [元算子：通过元算子实现自己的卷积层][3]

## Install

## 安装

我们提供了Docker安装方式，免去您配置环境，Docker安装方法如下：

We provide a Docker installation method to save you from configuring the environment. The Docker installation method is as follows:

```
# CPU only
docker run -it --network host jittor/jittor
# CPU and CUDA
docker run -it --network host jittor/jittor-cuda
```

关于Docker安装的详细教程，可以参考[Windows/Mac/Linux通过Docker安装计图](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-15-00-00-docker/)

Jittor is written in Python and C++. It requires a compiler for JIT compilation, Currently, we support four compilers:

Jittor使用Python和C++编写。 它需要用于即时编译的编译器。当前，我们支持三种编译器：

* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* CPU 编译器 （需要下列至少一个）
    - g++ （>=5.4.0）
    - clang （>=8.0）
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU 编译器（可选）
    - nvcc (>=10.0 for g++ 或者 >=10.2 for clang)

Jittor的环境要求如下:

* 操作系统: **Ubuntu** >= 16.04 (or **Windows** Subsystem of Linux)
* Python版本 >= 3.7
* C++编译器（g++ or clang）

注意：目前Jittor通过WSL的方式在Windows操作系统上运行，WSL的安装方法请参考[微软官网](https://docs.microsoft.com/en-us/windows/wsl/install-win10)，目前WSL尚不支持CUDA。

Jittor environment requirements:

* System: **Ubuntu** >= 16.04 (or **Windows** Subsystem of Linux)
* Python version >= 3.7
* C++ compiler(g++ or clang)

Note: Currently Jittor runs on the Windows operating system through WSL. For the installation method of WSL, please refer to [Microsoft official website](https://docs.microsoft.com/en-us/windows/wsl/install-win10). WSL does not yet support CUDA.

Jittor offers three ways to install: pip, script or manual.

Jittor 一共提供三种方式安装: pip安装, 一键脚本安装 和 手动安装.

## Pip 安装

## Pip install

如果您没有准备好环境，欢迎使用我们提供的一键安装脚本， 如果您已经装好编译器和对应版本的Python,我们强烈推荐您使用这种方法
(如果无法访问github, 可以通过jittor主页下载):

```bash
sudo apt install python3.7-dev libomp-dev
sudo python3.7 -m pip install git+https://github.com/Jittor/jittor.git
python3.7 -m jittor.test.test_example
```

如果测试运行通过,恭喜你已经安装完成.
jittor会自动在路径中寻找合适的编译器, 如果您希望手动指定编译器, 请使用环境变量 `cc_path` 和 `nvcc_path`(可选).

## 一键脚本安装
## single line script install

一键脚本安装会帮您安装好所需的编译器以及对应的Python版本.

We provide single line command for quick installation the latest version of Jittor(Ubuntu>=16.04):

我们提供能快速安装最新版本Jittor的单行命令（Ubuntu> = 16.04）：

```bash
# install with clang and cuda
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install.sh | with_clang=1 with_cuda=1 bash
# install with clang
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install.sh | with_clang=1 bash
# install with g++ and cuda
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install.sh | with_gcc=1 with_cuda=1 bash
# install with g++
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install.sh | with_gcc=1 bash
```
After execution, the script will show some environment variables you need to export.

执行后，脚本将显示一些需要导出的环境变量。

If you use Jittor for CPU computing, we strongly recommend clang(>=8.0) as the back-end compiler of Jittor. Because some customized optimizations will be enabled.

如果将Jittor用于CPU计算，则强烈建议使用clang（> = 8.0）作为Jittor的后端编译器。 因为Jittor会用到其中一些定制的优化。


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

整个框架是及时编译的。 让我们通过pip安装jittor

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

### Optional Step 5: Run full tests

### 可选步骤五：进行完整测试

To check the integrity of Jittor, you can run full tests.

要检查Jittor的完整性，您可以运行完整的测试。

```bash
python3.7 -m jittor.test -v
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
c.name('c')
print(c.name())
# Output: c
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



[1]: notebook/example.src.md	"example"
[2]: notebook/basics.src.md	"basics"
[3]: notebook/meta_op.src.md	"meta_op"
[4]: notebook/custom_op.src.md	"custom_op"
[5]: notebook/profiler.src.md	"profiler"
[1]: notebook/example.src.md	"示例"
[2]: notebook/basics.src.md	"基本概念"
[3]: notebook/meta_op.src.md	"元算子"
[4]: notebook/custom_op.src.md	"自定义算子"
[5]: notebook/profiler.src.md	"性能分析器"

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

## The Team

## 团队

Jittor is currently maintained by Dun Liang, Guo-Ye Yang, Guo-Wei Yang,  Wen-Yang Zhou and Meng-Hao Guo etc. from the [Tsinghua CSCG Group](https://cg.cs.tsinghua.edu.cn/). If you are also interested in Jittor and want to improve it, Please join us!

Jittor目前由来自[清华大学计算机图形学组](https://cg.cs.tsinghua.edu.cn/)的梁盾，杨国烨，杨国炜，周文洋和国孟昊等博士生维护。 如果您也对Jittor感兴趣并希望对其进行改进，请加入我们！

## License

## 版权声明

Jittor is Apache 2.0 licensed, as found in the LICENSE.txt file.

如LICENSE.txt文件中所示，Jittor使用Apache 2.0版权协议。
