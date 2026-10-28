# C++ Console Interface

The console interface embeds Jittor's Python runtime into C++17 programs. Its array bridge
lets typed C++ arrays enter Jittor operations with **one fewer Python-level copy**.

## Generate and compile the example

The two configuration commands must use the same Python interpreter:

```bash
python -m jittor_utils.config --cxx-example > example.cc
g++ -std=c++17 example.cc \
  $(python -m jittor_utils.config --include-flags --libs-flags --cxx-flags) \
  -o example
./example
```

The generated source includes the console header and initializes an embedded runtime:

```cpp
#include <pyjt/pyjt_console.h>
#include <iostream>

int main() {
    jittor::Console console;
    console.run("print('hello jt console', flush=True)");
}
```

Remember to flush when Python output interleaves with C++ streams.

## Exchange scalars and containers

`set<T>` / `get<T>` exchange values by name with the embedded namespace:

```cpp
console.set<int>("count", 1);
std::cout << console.get<int>("count") << std::endl;

std::vector<int> values{1, 2, 3, 4};
console.set("values", values);
auto result = console.get<std::vector<int>>("values");
```

Integer and floating-point scalars, strings, vectors, maps, and unordered_maps are
supported.

## Exchange Jittor arrays

`jittor::array<T, N>` records a fixed rank, shape, and its own data buffer:

```cpp
jittor::array<int, 2> input({2, 3}, {6, 5, 4, 3, 2, 1});
input(0, 0) = -1;
console.set_array("input", input);
console.run("output = input ** 2");
auto output = console.get_array<int, 2>("output");

std::cout << output.shape[0] << " " << output.shape[1] << std::endl;
```

The type and rank requested by `get_array<T, N>` **must match the value in the console
namespace**. The array interface provides `shape`, `data`, `size()`, `nbyte()`, `dtype()`,
`ndim()`, and indexed element access.

## Run a model

Arrays can be used directly as input to ordinary Python model code:

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

Related calls **share a single `Console` instance**. Creating the embedded interpreter
repeatedly breaks compile-cache reuse and complicates Python runtime lifetime management.
