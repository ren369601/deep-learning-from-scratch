# Week 1 / Day 1：需要掌握的 Python 语法

今天的目标不是系统学完整个 Python，而是先掌握写线性模型会用到的最小语法集合。

今天需要会：

```text
变量
数字计算
函数
list
for 循环
class
if __name__ == "__main__"
```

## 1. 变量

变量就是给一个值起名字。

```python
x = 3
w = 2
b = 1
```

这里：

```text
x 保存输入
w 保存权重
b 保存偏置
```

之后就可以用这些变量做计算：

```python
y = w * x + b
print(y)
```

输出：

```text
7
```

在 Python 里，不需要提前声明变量类型。

```python
x = 3       # int，整数
lr = 0.01   # float，小数
name = "model"  # str，字符串
```

## 2. 数字计算

常用运算符：

```python
a + b   # 加法
a - b   # 减法
a * b   # 乘法
a / b   # 除法
a ** b  # 幂运算
```

例如：

```python
x = 3
y = 2 * x + 1
print(y)
```

输出：

```text
7
```

深度学习里最常见的计算之一就是：

```python
y = w * x + b
```

## 3. print

`print()` 用来把结果显示出来。

```python
x = 3
y = 7

print(x)
print(y)
print(x, y)
```

输出：

```text
3
7
3 7
```

今天练习里我们会经常写：

```python
print(x, y)
```

表示同时打印输入和输出。

## 4. 函数

函数可以把一段计算封装起来，之后反复使用。

例如：

```python
def linear(x, w, b):
    return w * x + b
```

这就是一个最小线性模型。

调用函数：

```python
y = linear(3, 2, 1)
print(y)
```

输出：

```text
7
```

函数结构：

```python
def 函数名(参数1, 参数2):
    函数内部代码
    return 返回结果
```

注意：Python 用缩进表示代码属于函数内部。

正确：

```python
def predict(x):
    return 4 * x - 1
```

错误：

```python
def predict(x):
return 4 * x - 1
```

第二种会报错，因为 `return` 没有缩进。

## 5. list

`list` 用来保存一组数据。

```python
xs = [1, 2, 3, 4, 5]
```

这里 `xs` 里有 5 个输入。

你可以通过下标访问：

```python
print(xs[0])
print(xs[1])
```

输出：

```text
1
2
```

注意：Python 下标从 0 开始。

```text
xs[0] 是第 1 个元素
xs[1] 是第 2 个元素
```

不过今天更常用的是直接遍历整个 list。

## 6. for 循环

`for` 循环可以对 list 里的每个元素重复执行同一段代码。

```python
xs = [1, 2, 3, 4, 5]

for x in xs:
    y = 2 * x + 1
    print(x, y)
```

输出：

```text
1 3
2 5
3 7
4 9
5 11
```

这段代码的意思是：

```text
依次取出 xs 里的每个数字
把当前数字叫作 x
计算 y = 2x + 1
打印 x 和 y
```

深度学习里经常会批量处理很多数据，`for` 循环是最基础的批量处理方式。

## 7. 函数 + for 循环

我们可以把模型写成函数，再用循环处理多个输入。

```python
def linear(x, w, b):
    return w * x + b


xs = [1, 2, 3, 4, 5]

for x in xs:
    y = linear(x, 2, 1)
    print(x, y)
```

这样做的好处是：

```text
计算规则只写一次
多个输入可以重复使用
代码更清楚
```

以后神经网络也是类似思想：

```text
先定义模型
再把数据送进模型
```

## 8. class

`class` 用来定义一个对象。

今天我们可以用它来表示一个线性模型。

```python
class LinearModel:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def predict(self, x):
        return self.w * x + self.b
```

使用：

```python
model = LinearModel(4, -1)
y = model.predict(2)
print(y)
```

输出：

```text
7
```

这段代码的意思是：

```text
创建一个 LinearModel
把 w 设置成 4
把 b 设置成 -1
输入 x = 2
输出 y = 4 * 2 - 1 = 7
```

### __init__

`__init__` 是创建对象时自动执行的函数。

```python
def __init__(self, w, b):
    self.w = w
    self.b = b
```

当你写：

```python
model = LinearModel(4, -1)
```

Python 会自动调用：

```python
__init__(self, 4, -1)
```

于是：

```text
self.w = 4
self.b = -1
```

### self

`self` 表示当前这个对象自己。

比如：

```python
self.w
self.b
```

表示这个模型自己的 `w` 和 `b`。

如果创建两个模型：

```python
model1 = LinearModel(4, -1)
model2 = LinearModel(2, 3)
```

那么：

```text
model1 有自己的 w 和 b
model2 也有自己的 w 和 b
```

这也是以后 PyTorch 写模型时必须理解的语法。

## 9. if __name__ == "__main__"

你会在今天的代码文件最后看到：

```python
if __name__ == "__main__":
    demo_single_value()
    demo_list_values()
    practice()
    homework_function()
```

现在先简单理解成：

```text
当我直接运行这个 Python 文件时，就执行下面这些代码。
```

也就是运行：

```powershell
python week01\day01_python_linear_model.py
```

时，Python 会执行：

```python
demo_single_value()
demo_list_values()
practice()
homework_function()
```

这是一种常见写法，可以让代码结构更清楚。

## 10. 今天最重要的组合

今天真正要掌握的是这一套组合：

```python
def linear(x, w, b):
    return w * x + b


xs = [1, 2, 3, 4, 5]

for x in xs:
    y = linear(x, 2, 1)
    print(x, y)
```

你应该能解释：

```text
linear 是什么？
x 是什么？
w 是什么？
b 是什么？
xs 是什么？
for x in xs 是什么意思？
print(x, y) 会输出什么？
```

## 11. 今日语法小结

```text
变量：保存数据
函数：封装计算规则
list：保存一组数据
for：遍历一组数据
class：把参数和计算方法放在一起
self：表示当前对象自己
print：查看输出结果
```

用深度学习的语言说：

```text
函数 / class 可以表示模型
变量可以表示参数
list 可以表示一批输入数据
for 循环可以批量计算预测结果
```
