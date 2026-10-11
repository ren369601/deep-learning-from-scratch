# Week 4 / Day 6：训练与验证用到的 PyTorch 写法

## 1. 复用昨天的模型

```python
from day05_pytorch_mlp import TwoLayerMLP

model = TwoLayerMLP(2, 16, 2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
```

在 `week04` 目录运行 `python day06_train_validate.py`。分类模型输出 logits；不要在 Cross Entropy 前手动 Softmax。

## 2. 一次训练更新

```python
model.train()
optimizer.zero_grad()
logits = model(X_train)
loss = loss_fn(logits, y_train)
loss.backward()
optimizer.step()
```

这一段只使用训练集。`X_val`、`y_val` 不应参与 `backward()`。

## 3. 评估 loss 与 accuracy

```python
model.eval()
with torch.no_grad():
    logits = model(X_val)
    loss = loss_fn(logits, y_val).item()
    predicted = logits.argmax(dim=1)
    accuracy = (predicted == y_val).float().mean().item()
```

`argmax(dim=1)` 沿类别维选最大 logit。`float().mean()` 把布尔比较结果转为 0/1 并求平均。此处 `.item()` 只用于记录标量数值。

## 4. 验证参数没有变化

```python
before = [p.detach().clone() for p in model.parameters()]
_ = evaluate(model, X_val, y_val, loss_fn)
after = list(model.parameters())
print(all(torch.equal(a, b) for a, b in zip(before, after)))
```

这里的 `evaluate` 指当天练习文件提供的评估函数。验证只读模型参数；训练更新后参数才会改变。

## 5. 日志打印

```python
print(f"epoch={epoch:3d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
```

在 `optimizer.step()` 之后重新评估两个数据集，再打印，才能保证同一行的指标对应同一组参数。
