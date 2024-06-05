import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # Define Rosenbrock function gradient
# def rosenbrock_gradient(x, y):
#     return (-2 + 2 * x - 400 * x * (y - x**2), 200 * (y - x**2))

# # Set up the plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Generate grid for surface
# X = np.linspace(-2, 2, 400)
# Y = np.linspace(-1, 3, 400)
# X, Y = np.meshgrid(X, Y)
# Z = (1 - X)**2 + 100 * (Y - X**2)**2
# ax.plot_surface(X, Y, Z, rstride=10, cstride=10, alpha=0.4)

# # Simulate gradient descent
# x, y = -1, -1  # Initial condition
# lr = 0.001  # Learning rate
# history = [(x, y)]
# for _ in range(100):
#     grad_x, grad_y = rosenbrock_gradient(x, y)
#     new_x = x - lr * grad_x
#     new_y = y - lr * grad_y
#     new_x, new_y = np.clip(new_x, -2, 2), np.clip(new_y, -1, 3)  # Clip to prevent overflow
#     z = (1 - new_x)**2 + 100 * (new_y - new_x**2)**2
#     # Plot an arrow to the new point
#     ax.quiver(x, y, (1 - x)**2 + 100 * (y - x**2)**2, new_x - x, new_y - y, z - ((1 - x)**2 + 100 * (y - x**2)**2), color='r', length=0.1, normalize=True)
#     x, y = new_x, new_y
#     history.append((x, y))

# # Formatting
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()

# 创建图和轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建数据: 构造一个具有多个局部最小点的复杂曲面
X = np.arange(-2, 2.05, 0.05)
Y = np.arange(-2, 2.05, 0.05)
X, Y = np.meshgrid(X, Y)
# 修改曲面函数，添加一个鞍点
Z = np.sin(X) * np.cos(Y) + (X**2 - Y**2) / 8

# 创建图和轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制新的曲面
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# 隐藏坐标轴标签
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.show()
