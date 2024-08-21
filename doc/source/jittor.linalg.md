jittor.linalg
=====================

这里是Jittor的线性代数函数的API文档，您可以通过`from jittor import linalg`来获取该模块。

## 基本函数简介
#### 基本线性代数运算API
- linalg.inv(a)

  对a进行求逆运算

- linalg.pinv(a)

  对a进行广义求逆运算。该运算不要求原矩阵a可逆。

- linalg.slogdet(a)

  对a求取slogdet。会返回值以及符号。

- linalg.det(a)

  对a求行列式。

- linalg.solve(a,b)

  求解线性方程Ax=b的解。

#### 分解API
- linalg.cholesky(a)

  对a进行cholesky分解。

- linalg.qr(a)

  对a进行qr分解。

- linalg.svd

  对a进行奇异值分解。
####  特征值API
- linalg.eig(a)

  求取a的特征值以及特征向量。

- linalg.eigh(a)

  针对埃尔米特矩阵或者对称矩阵求特征值以及特征向量。
  

目前的linalg库支持

```eval_rst
.. automodule:: jittor.linalg
   :members:
   :undoc-members:
```

