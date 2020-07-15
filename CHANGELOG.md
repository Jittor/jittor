# CHANGELOG

### 计图 1.1.5.5

* 新增numpy code算子，现在可以直接使用numpy来自定义算子了，使用用例：

```python
import jittor as jt

def forward_code(np, data):
    a = data["inputs"][0]
    b = data["outputs"][0]
    np.add(a,a,out=b)

def backward_code(np, data):
    dout = data["dout"]
    out = data["outputs"][0]
    np.copyto(out, dout*2.0)

a = jt.random((5,1))
b = jt.numpy_code(
    a.shape,
    a.dtype,
    [a],
    forward_code,
    [backward_code],
)
```

* 新增 Function 模块，用户可以自定义反向传播了，使用用例：

```python
import jittor as jt
from jittor import Function

class MyFunc(Function):
    def execute(self, x, y):
        self.x = x
        self.y = y
        return x*y, x/y

    def grad(self, grad0, grad1):
        return grad0 * self.y, grad1 * self.x
a = jt.array(3.0)
b = jt.array(4.0)
func = MyFunc()
c,d = func(a, b)
da, db = jt.grad(c+d*3, [a, b])
assert da.data == 4
assert db.data == 9
```

* 新增 no_grad scope, 在这个scope中创建的所有变量都会停止梯度：

```python
import jittor as jt

with jt.no_grad():
    ...
```

* 新增 bmm（batch matrix multiply） 支持:

```python
import jittor as jt
from jittor import nn

batch, n, m, k = 100, 5, 6, 7

a = jt.random((batch, n, m))
b = jt.random((batch, m, k))
c = nn.bmm(a, b)
```

* 修复 unsqueeze