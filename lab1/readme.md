
#  Lab 1 – Intro to PyTorch (Part 1)

This lab introduces the basics of **PyTorch**, focusing on **tensors**, **computations**, **neural network layers**, and **gradient-based optimization**. These foundations are essential before moving into sequence modeling and RNNs later in the course.

##  Topics Covered

### 1. **Tensors in PyTorch**

* Scalars, vectors, matrices, and higher-dimensional tensors.
* Inspecting tensor properties:

  * `.ndim` → number of dimensions
  * `.shape` → size of each dimension
* Indexing and slicing subtensors.
* **Example:**

  ```python
  # 4D tensor: 10 RGB images, 256x256 each
  images = torch.zeros(10, 3, 256, 256)
  print(images.shape)  # torch.Size([10, 3, 256, 256])
  ```

---

### 2. **Tensor Computations (Computation Graphs)**

* PyTorch builds a **computation graph** of operations on tensors.
* Each operation (add, multiply, matmul) is a node in the graph.
* Example function:

  ```python
  def func(a, b):
      c = a + b
      d = a * b
      e = torch.matmul(c, d.T)
      return e
  ```

---

### 3. **Neural Networks Basics**

* Started from the **perceptron**:

  $$
  z = xW + b,\quad y = \sigma(z)
  $$
* Defined layers and models in three ways:

  1. **Custom Dense Layer** by subclassing `nn.Module`.
  2. **Sequential API** for quick prototyping.
  3. **Subclassing with extra flexibility** (e.g., identity mode).

---

### 4. **Custom Behavior**

* Example: `LinearButSometimesIdentity`

  * Normal mode: apply linear layer.
  * Identity mode: return inputs unchanged.

  ```python
  def forward(self, inputs, isidentity=False):
      if isidentity:
          return inputs
      return self.linear(inputs)
  ```

---

### 5. **Automatic Differentiation (`autograd`)**
Automatic differentiation is PyTorch’s way of automatically calculating gradients for any function you define, so you can train neural networks efficiently.
* PyTorch tracks computations automatically.
* Use `.backward()` to compute gradients.
* Example:

  ```python
  x = torch.tensor(3.0, requires_grad=True)
  y = x ** 2
  y.backward()
  print(x.grad)  # tensor(6.)
  ```

---

### 6. **Gradient Descent Optimization**

* Minimized a simple loss function:

  $$
  L = (x - x_f)^2
  $$
* Used **gradient descent** to iteratively update `x` toward target `x_f = 4`.
* Showed how loss decreases and `x` converges to the target.

```python
for i in range(500):
    x = torch.tensor([x], requires_grad=True)
    loss = (x - x_f) ** 2
    loss.backward()
    x = x.item() - learning_rate * x.grad
```

---

### Points to remember

* **Tensors** are the fundamental data structure in PyTorch.
* PyTorch builds a **computation graph** that supports **automatic differentiation**.
* Neural networks can be built with:

  * `nn.Sequential` for stacking layers.
  * `nn.Module` subclassing for custom architectures.
* **Training = forward pass → loss → backward pass → gradient descent update.**
