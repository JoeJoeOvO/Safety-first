import triangle as tr
import matplotlib.pyplot as plt

# 定义点集和约束边
points = [(0, 0), (1, 0), (1, 1), (0, 1)]  # 正方形四个顶点
segments = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 正方形的四条边（约束）

# 构建约束Delaunay三角剖分
A = dict(vertices=points, segments=segments)
B = tr.triangulate(A, 'p')  # 'p' 表示约束Delaunay

# 可视化
tr.compare(plt, A, B)
plt.show()