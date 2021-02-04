# Jittor: 即时编译深度学习框架


[快速开始](#快速开始) | [安装](#安装) | [教程](#教程)


Jittor 是一个基于即时编译和元算子的高性能深度学习框架，整个框架在即时编译的同时，还集成了强大的Op编译器和调优器，为您的模型生成定制化的高性能代码。Jittor还包含了丰富的高性能模型库，涵盖范围包括：图像识别，检测，分割，生成，可微渲染，几何学习，强化学习等等。


Jittor前端语言为Python。前端使用了模块化和动态图执行的设计，这是目前最主流的深度学习框架接口设计。后端则使用高性能语言编写，如CUDA，C++。


相关链接：
*  [Jittor官网](https://cg.cs.tsinghua.edu.cn/jittor/)
*  [Jittor教程](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/)
*  [Jittor模型库](https://cg.cs.tsinghua.edu.cn/jittor/resources/)
*  [Jittor文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)
*  [Github](https://github.com/jittor/jittor)， [Gitee](https://gitee.com/jittor/jittor)



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



## 大纲

- [快速开始](#快速开始)
- [安装](#安装)
- [教程](#教程)
- [贡献](#贡献)
- [团队](#团队)
- [版权声明](#版权声明)


## 快速开始


我们提供了一些jupyterr notebooks来帮助您快速入门Jittor。

- [示例：模型定义与训练][1]
- [基础：Op, Var][2]
- [元算子：通过元算子实现自己的卷积层][3]


## 安装


Jittor框架对环境要求如下:


* 操作系统: **Ubuntu** >= 16.04 或 **Windows Subsystem of Linux（WSL）**
* Python：版本 >= 3.7
* C++编译器 （需要下列至少一个）
    - g++ （>=5.4.0）
    - clang （>=8.0）
* GPU 编译器（可选）：nvcc >=10.0
* GPU 加速库（可选）：cudnn-dev (cudnn开发版, 推荐使用tar安装方法，[参考链接](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))

如果您不希望手动配置环境，我们推荐使用 Docker 进行安装。
除此之外，您还可以使用 pip 安装和手动安装。

注意：目前Jittor通过WSL的方式在Windows操作系统上运行，WSL的安装方法请参考[微软官网](https://docs.microsoft.com/en-us/windows/wsl/install-win10)，WSL版本目前尚不支持CUDA。

Jittor 提供了三种安装方法：docker，pip和手动安装：









## Docker 安装

我们提供了Docker安装方式，免去您配置环境，Docker安装方法如下：


```
# CPU only(Linux)
docker run -it --network host jittor/jittor
# CPU and CUDA(Linux)
docker run -it --network host --gpus all jittor/jittor-cuda
# CPU only(Mac and Windows)
docker run -it -p 8888:8888 jittor/jittor
```

关于Docker安装的详细教程，可以参考[Windows/Mac/Linux通过Docker安装计图](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-15-00-00-docker/)

## Pip 安装


如果您没有准备好环境，或者使用的不是Ubuntu操作系统， 推荐使用**docker安装**， 如果您已经装好编译器和对应版本的Python,我们强烈推荐您使用这种方法
(如果无法访问github, 可以通过jittor主页下载):

```bash
sudo apt install python3.7-dev libomp-dev
python3.7 -m pip install jittor
# or install from github(latest version)
# python3.7 -m pip install git+https://github.com/Jittor/jittor.git
python3.7 -m jittor.test.test_example
```

如果测试运行通过,恭喜你已经安装完成.
jittor会自动在路径中寻找合适的编译器, 如果您希望手动指定编译器, 请使用环境变量 `cc_path` 和 `nvcc_path`(可选).

## 手动安装


我们将逐步演示如何在Ubuntu 16.04中安装Jittor，其他Linux发行版可能可以使用类似的命令。


### 步骤一：选择您的后端编译器

```bash
# g++
sudo apt install g++ build-essential libomp-dev

# OR clang++-8
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install_llvm.sh > /tmp/llvm.sh
bash /tmp/llvm.sh 8
```

### 步骤二：安装Python和python-dev


Jittor需要python的版本>=3.7。

```bash
sudo apt install python3.7 python3.7-dev
```


### 步骤三：运行Jittor


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

如果通过了测试，那么您的Jittor已经准备就绪。


### 可选步骤四：启用CUDA


在Jittor中使用CUDA非常简单，只需设置环境值`nvcc_path`

```bash
# replace this var with your nvcc location 
export nvcc_path="/usr/local/cuda/bin/nvcc" 
# run a simple cuda test
python3.7 -m jittor.test.test_cuda 
```

如果测试通过，则可以通过设置`use_cuda`标识符在Jittor中启用CUDA。

```python
import jittor as jt
jt.flags.use_cuda = 1
```


### 可选步骤五：测试训练Resnet18


要检查Jittor的完整性，您可以运行Resnet18训练测试。需要注意的是，这个测试需要6G显存。

```bash
python3.7 -m jittor.test.test_resnet
```

如果这些测试失败，请为我们报告错误，我们十分欢迎您为Jittor做出贡献^ _ ^


## 教程


在教程部分，我们将简要解释Jittor的基本概念。


要使用Jittor训练模型，您需要了解两个主要概念：

* Var：Jittor的基本数据类型
* Operations：Jittor的算子与numpy类似


### 数据类型


首先，让我们开始使用Var。Var是jittor的基本数据类型，为了运算更加高效Jittor中的计算过程是异步的。 如果要访问数据，可以使用`Var.data`进行同步数据访问。

```python
import jittor as jt
a = jt.float32([1,2,3])
print (a)
print (a.data)
# Output: float32[3,]
# Output: [ 1. 2. 3.]
```


此外我们可以给变量起一个名字。

```python
a.name('a')
print(a.name())
# Output: a
```


### 数据运算


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

除此之外，我们使用的所有算子`jt.xxx(Var,...)`都具有别名`Var.xxx(...)`。 例如：

```python
c.max() # alias of jt.max(c)
c.add(a) # alias of jt.add(c, a)
c.min(keepdims=True) # alias of jt.min(c, keepdims=True)
```


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

### 更多教程


如果您想进一步了解Jittor，请查看以下notebooks：

* 快速开始
    * [示例：模型定义与训练][1]
    * [基本概念：Op, Var][2]
    * [元算子：通过元算子实现自己的卷积层][3]
* 进阶
    * [自定义算子：使用C ++和CUDA编写您的算子，并其进行即时编译][4]
    * [性能分析器：分析您的模型][5]
    * Jtune：性能调优工具



[1]: notebook/example.src.md	"示例"
[2]: notebook/basics.src.md	"基本概念"
[3]: notebook/meta_op.src.md	"元算子"
[4]: notebook/custom_op.src.md	"自定义算子"
[5]: notebook/profiler.src.md	"性能分析器"


这些notebooks可以通过python3.7 -m jittor.notebook在您自己的计算机中运行。


## 贡献


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




## 联系我们

官方主页： http://cg.cs.tsinghua.edu.cn/jittor/

电子邮件：jittor@qq.com

提出issue：https://github.com/Jittor/jittor/issues





QQ 群：761222083



## 团队


Jittor目前由[清华大学计算机图形学组](https://cg.cs.tsinghua.edu.cn/)维护。 如果您也对Jittor感兴趣并希望对其进行改进，请加入我们！


## 引用

```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```


## 版权声明


如LICENSE.txt文件中所示，Jittor使用Apache 2.0版权协议。
