# C++ console interface

The console interface embeds Jittor's Python runtime in a C++17 program. Its
array bridge avoids an extra Python-level copy when moving typed C++ arrays into
Jittor operations.

## Generate and compile the example

Use the active Python interpreter for both configuration commands:

```bash
python -m jittor_utils.config --cxx-example > example.cc
g++ -std=c++17 example.cc \
  $(python -m jittor_utils.config --include-flags --libs-flags --cxx-flags) \
  -o example
./example
```

The generated source includes the console header and initializes one embedded
runtime:

```cpp
#include <pyjt/pyjt_console.h>
#include <iostream>

int main() {
    jittor::Console console;
    console.run("print('hello jt console', flush=True)");
}
```

Flush Python output when it is interleaved with C++ streams.

## Exchange scalar and container values

`set<T>` and `get<T>` exchange named values with the embedded namespace:

```cpp
console.set<int>("count", 1);
std::cout << console.get<int>("count") << std::endl;

std::vector<int> values{1, 2, 3, 4};
console.set("values", values);
auto result = console.get<std::vector<int>>("values");
```

Supported families include integer and floating-point scalars, strings,
vectors, maps, and unordered maps.

## Exchange Jittor arrays

`jittor::array<T, N>` records a fixed rank, shape, and owned data buffer:

```cpp
jittor::array<int, 2> input({2, 3}, {6, 5, 4, 3, 2, 1});
input(0, 0) = -1;
console.set_array("input", input);
console.run("output = input ** 2");
auto output = console.get_array<int, 2>("output");

std::cout << output.shape[0] << " " << output.shape[1] << std::endl;
```

The requested type and rank in `get_array<T, N>` must match the value in the
console namespace. The array interface provides `shape`, `data`, `size()`,
`nbyte()`, `dtype()`, `ndim()`, and indexed element access.

## Run a model

Arrays can be inputs to normal Python model code:

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

Keep one `Console` instance for related calls. Repeatedly creating embedded
interpreters defeats compilation-cache reuse and complicates Python runtime
lifetime management.
