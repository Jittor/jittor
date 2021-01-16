jittor.console
=====================

这里是Jittor的console api文档，console功能主要面向c/c++, 方便c++用户通过console使用jittor，jittor console 优化了
c++数组和jittor内核之间的数据传输，减少了python额外开销，是通过c++使用jittor的高性能接口。

该功能要求 jittor版本大于1.2.2.17, 编译器支持c++17。

## 简单教程

我们提供了一个完整的教程，用户可以通过如下几行命令编译运行：

```bash
# 生成c++ example源代码文件
python3.7 -m jittor_utils.config --cxx-example > example.cc
# 调用g++编译example, 需要g++支持std=c++17
g++ example.cc $(python3.7 -m jittor_utils.config --include-flags --libs-flags --cxx-flags) -o example
# 运行example
./example
```

运行结果可能如下：
```bash
hello jt console
1
hello
1 2 3 4 
jt.Var([[-1  5  4]
 [ 3  2  1]], dtype=int32)
2 3
1 25 16 
9 4 1 
pred.shape 2 1000
```

用户可以打开 example.cc, 修改成所需的应用，接下来我们会为大家讲解 example.cc 中的细节。

打开example.cc, 我们可以看到如下代码：

```cpp
#include <pyjt/pyjt_console.h>
#include <iostream>

using namespace std;

int main() {
   ...
}
```

这里我们导入了使用 console 所需的头文件 `pyjt/pyjt_console.h`

接下来是jittor console的实例化， 并且使用python的print输出hello jt console：

```cpp
    jittor::Console console;
    // run python code in console
    console.run("print('hello jt console', flush=True)");
```

输出结果：

```
hello jt console
```

注意到这里我们在 python print的时候使用了flush keyword，这是为了让python的输出流和c++的输出流保持一致，
不会错乱。

接下来我们调用了 `console.set<T>(name, data)` 和 `console.get<T>(name)` 往 console 里面设置了一个int变量a，并且再从console里面取出来。

```cpp
    // set a python value: a = 1
    console.set<int>("a", 1);
    // get a python value
    cout << console.get<int>("a") << endl;
```

输出结果：

```
1
```

同样的方法，我们还设置了 `string` 和 `vector<int>`， 如下所示

```cpp
    // set a python string
    console.set<string>("b", "hello");
    cout << console.get<string>("b") << endl;

    // set a python array
    vector<int> x{1,2,3,4};
    console.set("x", x);
    auto x2 = console.get<std::vector<int>>("x");
    for (auto a : x2) cout << a << " "; cout << endl;
```

输出结果：

```
hello
1 2 3 4 
```

我们还可以往console里面设置jittor变量，这里我们使用了下面几个新的接口：

1. `jittor::array<T, NDIM>(shape, data)`: 这个接口创建了一个jittor的array，类型是`T`， 维度大小为`NDIM`， 形状为 `shape`， 注意shape的长度需要和`NDIM`保持一致，最后是传入的数据，可以是一个vector，也可以是一个指针。
2. `console.set_array(name, arr)`: 往console里面设置该jittor array， 名称为`name`。
3. `console.get<T, NDIM>(name)`: 从console里取出一个jittor array，类型为`T`，维度大小为`NDIM`，需要注意的是类型和维度大小必须和console中的变量匹配，否则会抛出异常。
4. `arr(i,j)`: 对jittor变量取值。
5. `arr.shape[i]`: 获取jittor变量的维度大小。

在这段代码中，我们首先创建了一个2x3的矩阵， 然后修改了矩阵中的值，随即设置到了python console里面，并且取出输出：

```cpp
    // set and get a jittor array
    jittor::array<int, 2> arr2({2,3}, {6,5,4,3,2,1});
    arr2(0,0) = -1;
    console.set_array("arr2", arr2);
    console.run("print(arr2, flush=True); arr3 = arr2**2;");
    auto arr3 = console.get_array<int, 2>("arr3");
    cout << arr3.shape[0] << ' ' << arr3.shape[1] << endl;
    for (int i=0; i<arr3.shape[0]; i++) {
        for (int j=0; j<arr3.shape[1]; j++)
            cout << arr3(i,j) << ' ';
        cout << endl;
    }
```

输出结果如下：

```
jt.Var([[-1  5  4]
 [ 3  2  1]], dtype=int32)
2 3
1 25 16 
9 4 1 
```

最后，我们演示了从`jittor.models`中导入`resnet`并且将结果从console中取出。

```cpp
    jittor::array<float, 4> input({2, 3, 224, 224});
    memset(input.data.get(), 0, input.nbyte());
    console.set_array("input", input);
    console.run(R"(
import jittor as jt
from jittor.models import resnet

model = resnet.resnet18()
pred = model(input)
    )");
    auto pred = console.get_array<float, 2>("pred");
    cout << "pred.shape " << pred.shape[0] << ' ' << pred.shape[1] << endl;
```

我们输出了取出的变量的形状，结果如下：

```
pred.shape 2 1000
```

## jittor array 接口一览

`jittor::array` 是 c++和jittor console交互的 array类型，他的定义如下：

```cpp

// T： 类型， N： 维度数量
template<class T, int N>
struct array {

// N维 形状大小
int64 shape[N];
// 数据指针
unique_ptr<T[]> data;

// 是否为浮点数
bool is_float();
// 是否为无符号类型
bool is_unsigned();
// 数组总大小，为shape数组累乘的结果
int64 size();
// 数组总比特数
int64 nbyte();
// 数据类型的字符串表示
string dtype();
// 维度数量， 同 N
int ndim();

// array 构造函数，shape为形状，数据未被初始化
array(const vector<int64>& shape);
// array 构造函数，shape为形状，数据从data指针拷贝初始化
array(const vector<int64>& shape, const T* data);
// array 构造函数，shape为形状，数据从data vector拷贝初始化
array(const vector<int64>& shape, const vector<T>& data);

T& operator()(...);

};
```

## Console 接口一览

console接口主要用于设置变量，取出变量，运行脚本， 三部分构成。

```cpp

struct Console {

// 运行代码接口
void run(const string& src);

// 设置变量名称为s， 值为data
template<class T>
void set(const string& s, const T& data);

// 获取变量名称为s
template<class T>
T get(const string& s)

// 设置 array 变量
void set_array(const string& s, const array<T,N>& data);

// 获取一个jittor array，类型为`T`，维度大小为`NDIM`，需要注意的是类型和维度大小必须和console中的变量匹配，否则会抛出异常。
void get_array<T,N>(const string& s);

};
```

其中 `get`，`set` 支持常见的c++类型有：

1. int， uint, int64, uint64, float, double
2. string
3. vector
4. map, unordered_map
