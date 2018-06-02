import numpy as np
import matplotlib.pyplot as plt


# 递归求差商
def get_diff_quo(xi, fi):
    if len(xi) > 2 and len(fi) > 2:
        return (get_diff_quo(xi[:len(xi) - 1], fi[:len(fi) - 1]) - get_diff_quo(xi[1:len(xi)], fi[1:len(fi)])) / float(
            xi[0] - xi[-1])
    return (fi[0] - fi[1]) / float(xi[0] - xi[1])


# 求w，使用闭包函数
def get_w(i, xi):
    def wi(x):
        result = 1.0
        for j in range(i):
            result *= (x - xi[j])
        return result

    return wi


# 做插值
def get_Newton(xi, fi):
    def Newton(x):
        result = fi[0]
        for i in range(2, len(xi)):
            result += (get_diff_quo(xi[:i], fi[:i]) * get_w(i - 1, xi)(x))
        return result

    return Newton


# 已知结点
xn = [i for i in range(-50, 50, 10)]
fn = [i ** 2 for i in xn]

# 插值函数
Nx = get_Newton(xn, fn)

# 测试用例
tmp_x = [i for i in range(-50, 51)]
tmp_y = [Nx(i) for i in tmp_x]

print(tmp_x)
print(tmp_y)

# 作图
plt.plot(xn, fn, 'r*')
plt.plot(tmp_x, tmp_y, 'b-')
plt.title('Newton Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
