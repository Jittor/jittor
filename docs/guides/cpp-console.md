# C++ 控制台接口

控制台接口把 Jittor 的 Python 运行时嵌入到 C++17 程序里。它的数组桥接让带类型的
C++ 数组进入 Jittor 运算时**少一次 Python 层拷贝**。

## 生成并编译示例

两条配置命令要用同一个 Python 解释器：

```bash
python -m jittor_utils.config --cxx-example > example.cc
g++ -std=c++17 example.cc \
  $(python -m jittor_utils.config --include-flags --libs-flags --cxx-flags) \
  -o example
./example
```

生成的源码包含控制台头文件并初始化一个嵌入运行时：

```cpp
#include <pyjt/pyjt_console.h>
#include <iostream>

int main() {
    jittor::Console console;
    console.run("print('hello jt console', flush=True)");
}
```

Python 输出与 C++ 流交错时记得 flush。

## 交换标量与容器

`set<T>` / `get<T>` 按名字与嵌入命名空间交换值：

```cpp
console.set<int>("count", 1);
std::cout << console.get<int>("count") << std::endl;

std::vector<int> values{1, 2, 3, 4};
console.set("values", values);
auto result = console.get<std::vector<int>>("values");
```

支持整数与浮点标量、字符串、vector、map 和 unordered_map。

## 交换 Jittor 数组

`jittor::array<T, N>` 记录固定的秩、形状和自有数据缓冲：

```cpp
jittor::array<int, 2> input({2, 3}, {6, 5, 4, 3, 2, 1});
input(0, 0) = -1;
console.set_array("input", input);
console.run("output = input ** 2");
auto output = console.get_array<int, 2>("output");

std::cout << output.shape[0] << " " << output.shape[1] << std::endl;
```

`get_array<T, N>` 请求的类型和秩**必须与控制台命名空间里的值一致**。数组接口提供
`shape`、`data`、`size()`、`nbyte()`、`dtype()`、`ndim()` 和按下标访问元素。

## 跑一个模型

数组可以直接作为普通 Python 模型代码的输入：

```cpp
jittor::array<float, 4> input({2, 3, 224, 224});
std::memset(input.data.get(), 0, input.nbyte());
console.set_array("input", input);
console.run(R"(
from jittor.models import resnet

model = resnet.resnet18()
prediction = model(input)
)");
auto prediction = console.get_array<float, 2>("prediction");
```

相关调用**共用一个 `Console` 实例**。反复创建嵌入解释器会破坏编译缓存复用，也让
Python 运行时的生命周期管理复杂化。
