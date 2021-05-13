# Jittor: a Just-in-time(JIT) deep learning framework

[Quickstart](#quickstart) | [Install](#install) | [Tutorial](#tutorial) | [Chinese](./README.cn.md)


Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model. Jittor also contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc. .


The front-end language is Python. Module Design and Dynamic Graph Execution is used in the front-end, which is the most popular design for deeplearning framework interface. The back-end is implemented by high performance language, such as CUDA,C++.


Related Links:
*  [Jittor Website](https://cg.cs.tsinghua.edu.cn/jittor/)
*  [Jittor Tutorials](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/)
*  [Jittor Models](https://cg.cs.tsinghua.edu.cn/jittor/resources/)
*  [Jittor Documents](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)
*  [Github](https://github.com/jittor/jittor), [Gitee](https://gitee.com/jittor/jittor)



The following example shows how to model a two-layer neural network step by step and train from scratch In a few lines of Python code.


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



## Quickstart


We provide some jupyter notebooks to help you quick start with Jittor.


- [Example: Model definition and training][1]
- [Basics: Op, Var][2]
- [Meta-operator: Implement your own convolution with Meta-operator][3]

## Install









Jittor environment requirements:

* System: **Ubuntu** >= 16.04 (or **Windows** Subsystem of Linux)
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU library: cudnn-dev (recommend tar file installation, [reference link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))



Note: Currently Jittor runs on the Windows operating system through WSL. For the installation method of WSL, please refer to [Microsoft official website](https://docs.microsoft.com/en-us/windows/wsl/install-win10). WSL does not yet support CUDA.

Jittor offers three ways to install: docker, pip, or manual.


## Docker Install



We provide a Docker installation method to save you from configuring the environment. The Docker installation method is as follows:

```
# CPU only(Linux)
docker run -it --network host jittor/jittor
# CPU and CUDA(Linux)
docker run -it --network host --gpus all jittor/jittor-cuda
# CPU only(Mac and Windows)
docker run -it -p 8888:8888 jittor/jittor
```



## Pip install


```bash
sudo apt install python3.7-dev libomp-dev
python3.7 -m pip install jittor
# or install from github(latest version)
# python3.7 -m pip install git+https://github.com/Jittor/jittor.git
python3.7 -m jittor.test.test_example
```


## manual install

We will show how to install Jittor in Ubuntu 16.04 step by step, Other Linux distributions may have similar commands.


### Step 1: Choose your back-end compiler


```bash
# g++
sudo apt install g++ build-essential libomp-dev

# OR clang++-8
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install_llvm.sh > /tmp/llvm.sh
bash /tmp/llvm.sh 8
```
### Step 2: Install Python and python-dev


Jittor need python version >= 3.7.


```bash
sudo apt install python3.7 python3.7-dev
```

### Step 3: Run Jittor


The whole framework is compiled Just-in-time. Let's install jittor via pip


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


### Optional Step 4: Enable CUDA


Using CUDA in Jittor is very simple, Just setup environment value `nvcc_path`


```bash
# replace this var with your nvcc location 
export nvcc_path="/usr/local/cuda/bin/nvcc" 
# run a simple cuda test
python3.7 -m jittor.test.test_cuda 
```
if the test is passed, your can use Jittor with CUDA by setting `use_cuda` flag.


```python
import jittor as jt
jt.flags.use_cuda = 1
```

### Optional Step 5: Test Resnet18 training


To check the integrity of Jittor, you can run Resnet18 training test. Note: 6G GPU RAM is requires in this test.


```bash
python3.7 -m jittor.test.test_resnet
```
if those tests are failed, please report bugs for us, and feel free to contribute ^_^


## Tutorial


In the tutorial section, we will briefly explain the basic concept of Jittor.


To train your model with Jittor, there are only three main concepts you need to know:


* Var: basic data type of jittor
* Operations: Jittor'op is simular with numpy

### Var


First, let's get started with Var. Var is the basic data type of jittor. Computation process in Jittor is asynchronous for optimization. If you want to access the data, `Var.data` can be used for synchronous data accessing.


```python
import jittor as jt
a = jt.float32([1,2,3])
print (a)
print (a.data)
# Output: float32[3,]
# Output: [ 1. 2. 3.]
```

And we can give the variable a name.


```python
a.name('a')
print(a.name())
# Output: a
```

### Operations


Jittor'op is simular with numpy. Let's try some operations. We create Var `a` and `b` via operation `jt.float32`, and add them. Printing those variables shows they have the same shape and dtype.


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


```python
c.max() # alias of jt.max(c)
c.add(a) # alias of jt.add(c, a)
c.min(keepdims=True) # alias of jt.min(c, keepdims=True)
```

if you want to know all the operation which Jittor supports. try `help(jt.ops)`. All the operation you found in `jt.ops.xxx`, can be used via alias `jt.xxx`.


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


If you want to know more about Jittor, please check out the notebooks below:


* Quickstart
    - [Example: Model definition and training][1]
    - [Basics: Op, Var][2]
    - [Meta-operator: Implement your own convolution with Meta-operator][3]
* Advanced
    - [Custom Op: write your operator with C++ and CUDA and JIT compile it][4]
    - [Profiler: Profiling your model][5]
    - Jtune: Tool for performance tuning



[1]: notebook/example.src.md	"example"
[2]: notebook/basics.src.md	"basics"
[3]: notebook/meta_op.src.md	"meta_op"
[4]: notebook/custom_op.src.md	"custom_op"
[5]: notebook/profiler.src.md	"profiler"

Those notebooks can be started in your own computer by `python3.7 -m jittor.notebook`


## Contributing


Jittor is still young. It may contain bugs and issues. Please report them in our bug track system. Contributions are welcome. Besides, if you have any ideas about Jittor, please let us know.




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





Website: http://cg.cs.tsinghua.edu.cn/jittor/

Email: jittor@qq.com

File an issue: https://github.com/Jittor/jittor/issues

QQ Group: 761222083


<img src="https://cg.cs.tsinghua.edu.cn/jittor/images/news/2020-12-8-21-19-1_2_2/fig4.png" width="200"/>

## The Team


Jittor is currently maintained by the [Tsinghua CSCG Group](https://cg.cs.tsinghua.edu.cn/). If you are also interested in Jittor and want to improve it, Please join us!


## Citation


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

## License


Jittor is Apache 2.0 licensed, as found in the LICENSE.txt file.

