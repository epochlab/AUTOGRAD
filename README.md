# AUTOGRAD

**Project ID:**  U8bTnBcp

<p align="center">
  <img src="https://github.com/epochlab/AUTOGRAD/blob/main/sample.png">
</p>

--------------------------------------------------------------------

#### A simple deep learning framework.
Abstract: *A scalar-valued autograd engine which tracks values, gradients and executed operations over a dynamically built DAG (directed acyclical graph).*

### Example

```python
from autograd.engine import Value

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
d = a * b; d.label='d'
e = d + c; e.label='e'
f = Value(-2.0, label='f')
L = e * f; L.label='L'

print(f'{L.data:.4f}') # prints -8.0, the outcome of this forward pass
L.backward()
print(f'{a.grad:.4f}') # prints 6.0, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints -4.0, i.e. the numerical value of dg/db
```

### Requirements
- Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
- 64-bit Python 3.7.9 installation.

### Acknowledgments
[Micrograd](https://github.com/karpathy/micrograd) (2020)<br />
[Tinygrad](https://github.com/geohot/tinygrad) (2022)
