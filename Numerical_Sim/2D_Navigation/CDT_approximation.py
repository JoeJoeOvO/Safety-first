import triangle as tr
import matplotlib.pyplot as plt
import numpy as np

from obstacles_generator import *


def constrained_delaunay(polygons):
    """
    对多个无孔多边形进行约束Delaunay三角剖分(CDT)并绘制结果
    
    参数:
        polygons: list of numpy arrays, 每个数组表示一个多边形的顶点坐标[N,2]
                 顶点需按顺时针或逆时针顺序排列
    
    返回:
        dict: 包含顶点和三角形信息的字典
    """
    # 合并所有顶点并记录约束边
    vertices = np.vstack(polygons)  # 合并所有顶点
    segments = []
    vertex_offset = 0
    
    for poly in polygons:
        n = len(poly)
        # 添加当前多边形的约束边 (连接相邻顶点)
        segments.extend([(vertex_offset + i, vertex_offset + (i + 1) % n) 
                        for i in range(n)])
        vertex_offset += n
    
    # 构建输入数据结构
    mesh_data = {
        'vertices': vertices,
        'segments': np.array(segments)
    }
    
    # 执行约束Delaunay三角剖分 (p: 强制约束边, D: Delaunay性质)
    result = tr.triangulate(mesh_data, 'pDa0.5')
    
    # 绘制结果
    plt.figure(figsize=(10, 8))
    
    # 1. 绘制原始多边形边界 (红色)
    for poly in polygons:
        closed_poly = np.vstack([poly, poly[0]])  # 闭合多边形
        plt.plot(closed_poly[:, 0], closed_poly[:, 1], 'r-', linewidth=2, label='Input Polygon' if not plt.gca().get_legend() else "")
    
    # 2. 绘制三角剖分结果 (蓝色)
    plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
               result['triangles'], 'b-', alpha=0.5, label='CDT Triangles')
    
    # 3. 标记所有顶点 (黑点)
    plt.plot(result['vertices'][:, 0], result['vertices'][:, 1], 'ko', markersize=4, label='Vertices')
    
    plt.title('Constrained Delaunay Triangulation (No Holes)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.show()
    
    return result


def CDT_circumcircles(cdt_result):
    """
    计算并绘制CDT结果中每个三角形的外接圆
    
    参数:
        cdt_result: dict, cdt_triangulation_and_plot()函数的返回结果
                   必须包含'vertices'和'triangles'字段
    
    返回:
        list: 每个外接圆的(x, y, r)元组列表
    """
    vertices = cdt_result['vertices']
    triangles = cdt_result['triangles']
    circumcircles = []
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. 绘制原始三角网格
    ax.triplot(vertices[:, 0], vertices[:, 1], triangles, 'b-', alpha=0.3)
    ax.plot(vertices[:, 0], vertices[:, 1], 'ko', markersize=4)
    
    # 2. 计算并绘制每个三角形的外接圆
    for tri in triangles:
        A, B, C = vertices[tri]
        
        # 计算外接圆圆心和半径
        D = 2 * (A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
        if abs(D) < 1e-12:  # 处理共线情况
            continue
            
        Ux = ((A[0]**2 + A[1]**2)*(B[1]-C[1]) + 
              (B[0]**2 + B[1]**2)*(C[1]-A[1]) + 
              (C[0]**2 + C[1]**2)*(A[1]-B[1])) / D
        Uy = ((A[0]**2 + A[1]**2)*(C[0]-B[0]) + 
              (B[0]**2 + B[1]**2)*(A[0]-C[0]) + 
              (C[0]**2 + C[1]**2)*(B[0]-A[0])) / D
        r = np.sqrt((A[0]-Ux)**2 + (A[1]-Uy)**2)
        
        circumcircles.append((Ux, Uy, r))
        
        # 绘制外接圆
        circle = Circle((Ux, Uy), r, fill=False, color='green', alpha=0.5, linewidth=0.8)
        ax.add_patch(circle)
        ax.plot(Ux, Uy, 'g.', markersize=3)  # 圆心
    
    # 3. 可视化设置
    ax.set_title('Circumcircles of CDT Triangles')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True)
    
    # 添加图例
    blue_line = plt.Line2D([], [], color='blue', label='Delaunay Triangles')
    green_circle = plt.Line2D([], [], color='green', marker='o', linestyle='None',
                            markersize=5, label='Circumcircles')
    black_dot = plt.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=5, label='Vertices')
    ax.legend(handles=[blue_line, green_circle, black_dot])
    
    plt.show()
    
    return circumcircles

def reduce_circles(circles, d_bias):
    """
    通过合并覆盖圆缩减圆的数量
    
    参数:
        circles: list of tuples, 每个元组为 (x, y, r)
        d_bias: float, 合并阈值常数
    
    返回:
        list: 缩减后的圆列表
    """
    if not circles:
        return []
    
    # 按半径降序排序，优先处理大圆
    circles_sorted = sorted(circles, key=lambda c: -c[2])
    reduced = []
    
    while circles_sorted:
        # 取出当前最大的圆
        current = circles_sorted.pop(0)
        x1, y1, r1 = current
        to_remove = []
        
        # 检查剩余圆是否能被当前圆覆盖
        for i, (x2, y2, r2) in enumerate(circles_sorted):
            d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if d <= r1 - r2 + d_bias:
                to_remove.append(i)  # 标记待移除的小圆
        
        # 从后向前删除，避免索引错乱
        for i in sorted(to_remove, reverse=True):
            circles_sorted.pop(i)
        
        # 扩大当前圆的半径并保留
        reduced.append((x1, y1, r1 + d_bias))
    
    return reduced,len(reduced)

def plot_circle_reduction(circles, d_bias):
    """
    1. 缩减圆的数量
    2. 并排绘制原始圆和缩减后的圆进行对比
    
    参数:
        circles: list of tuples, 每个元组为 (x, y, r)
        d_bias: float, 合并阈值常数
    """
    # 缩减圆的数量
    reduced_circles,_ = reduce_circles(circles, d_bias=0.2)
    
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制原始圆
    for x, y, r in circles:
        circle = Circle((x, y), r, fill=False, color='blue', alpha=0.5, linestyle='-', linewidth=1)
        ax1.add_patch(circle)
        ax1.plot(x, y, 'b.', markersize=5)  # 圆心
    
    ax1.set_title(f'Original Circles (n={len(circles)})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.autoscale_view()
    
    # 绘制缩减后的圆
    for x, y, r in reduced_circles:
        circle = Circle((x, y), r, fill=False, color='red', alpha=0.5, linestyle='--', linewidth=1.5)
        ax2.add_patch(circle)
        ax2.plot(x, y, 'r.', markersize=5)  # 圆心
    
    ax2.set_title(f'Reduced Circles (n={len(reduced_circles)})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.autoscale_view()
    
    plt.tight_layout()
    plt.show()
    return reduced_circles,len(reduced_circles)


MAP_X_RANGE = (0, 20)
MAP_Y_RANGE = (0, 20)    
polygons = generate_shapes(
    types='polytope',
    num_shapes=3,
    x_range=MAP_X_RANGE,
    y_range=MAP_Y_RANGE,
    vertex_range=(5, 20),
    size_limit=(1, 2))

# print(polygons)
cdt_results=constrained_delaunay(polygons)
circles=CDT_circumcircles(cdt_results)


# polygon1 = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])  # 矩形
# polygon2 = np.array([[3, 1], [5, 1], [4, 3]])          # 三角形
# polygon3 = np.array([[6, 0], [8, 0], [7, 2], [6.5, 1]]) # 四边形
# # 调用函数
# cdt_results = constrained_delaunay([polygon1, polygon2, polygon3])
# circles=CDT_circumcircles(cdt_results)

num_circles_temp=len(circles)
num_circles=num_circles_temp+1
while num_circles_temp<num_circles:
    # print(num_circles_temp)
    num_circles=num_circles_temp
    circles,num_circles_temp=plot_circle_reduction(circles,0.2)

# plot_circle_reduction(circles,d_bias=0.2)
