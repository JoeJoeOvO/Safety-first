import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from matplotlib.patches import Circle, Polygon

def generate_shapes(types: str, 
                   num_shapes: int, 
                   x_range: Tuple[float, float], 
                   y_range: Tuple[float, float],
                   radius_range: Tuple[float, float] = None,
                   vertex_range: Tuple[int, int] = None) -> Union[List[Tuple[float, float, float]], List[List[Tuple[float, float]]]]:
    """
    在地图范围内随机生成图形
    
    参数:
        types: 图形类型，'circle'或'polytope'
        num_shapes: 要生成的图形数量
        x_range: x轴范围 (xmin, xmax)
        y_range: y轴范围 (ymin, ymax)
        radius_range: 可选，圆的半径范围 (rmin, rmax)
        vertex_range: 可选，多边形的顶点数量范围 (vmin, vmax)
        
    返回:
        对于圆形: 返回圆心和半径列表 [(x1, y1, r1), (x2, y2, r2), ...]
        对于多边形: 返回顶点坐标列表 [[(x1, y1), (x2, y2), ...], [...]]
    """
    if types not in ['circle', 'polytope']:
        raise ValueError("types参数必须是'circle'或'polytope'")
    
    if types == 'circle' and radius_range is None:
        raise ValueError("生成圆形需要提供radius_range参数")
    
    if types == 'polytope' and vertex_range is None:
        raise ValueError("生成多边形需要提供vertex_range参数")
    
    xmin, xmax = x_range
    ymin, ymax = y_range
    
    if types == 'circle':
        rmin, rmax = radius_range
        shapes = []
        for _ in range(num_shapes):
            # 随机生成圆心和半径
            x = np.random.uniform(xmin + rmax, xmax - rmax)  # 确保圆不会超出边界
            y = np.random.uniform(ymin + rmax, ymax - rmax)
            r = np.random.uniform(rmin, rmax)
            shapes.append((x, y, r))
        return shapes
    
    elif types == 'polytope':
        vmin, vmax = vertex_range
        shapes = []
        for _ in range(num_shapes):
            # 随机确定顶点数量
            num_vertices = np.random.randint(vmin, vmax + 1)
            
            # 生成多边形顶点
            vertices = []
            for _ in range(num_vertices):
                x = np.random.uniform(xmin, xmax)
                y = np.random.uniform(ymin, ymax)
                vertices.append((x, y))
            
            # 按极角排序，确保多边形不自交
            center = np.mean(vertices, axis=0)
            angles = np.arctan2(
                np.array([v[1] - center[1] for v in vertices]),
                np.array([v[0] - center[0] for v in vertices])
            )
            vertices = [v for _, v in sorted(zip(angles, vertices))]
            
            shapes.append(vertices)
        return shapes

def plot_shapes(shapes: Union[List[Tuple[float, float, float]], List[List[Tuple[float, float]]]], 
                x_range: Tuple[float, float], 
                y_range: Tuple[float, float]):
    """
    可视化生成的图形
    
    参数:
        shapes: generate_shapes函数返回的结果
        x_range: x轴范围 (xmin, xmax)
        y_range: y轴范围 (ymin, ymax)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制图形
    if isinstance(shapes[0], tuple):  # 圆形
        for x, y, r in shapes:
            circle = Circle((x, y), r, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(circle)
    else:  # 多边形
        for vertices in shapes:
            polygon = Polygon(vertices, closed=True, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(polygon)
    
    # 设置图形范围
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_aspect('equal')
    ax.grid(True)
    plt.title("Random Obstacles")
    plt.xlabel("Axis X")
    plt.ylabel("Axis Y")
    plt.show()

# 示例用法
if __name__ == "__main__":
    # 定义地图范围
    MAP_X_RANGE = (0, 10)
    MAP_Y_RANGE = (0, 10)
    
    print("=== 示例1: 生成5个圆形 ===")
    circles = generate_shapes(
        types='circle',
        num_shapes=5,
        x_range=MAP_X_RANGE,
        y_range=MAP_Y_RANGE,
        radius_range=(0.2, 1.5)
    )
    print("生成的圆形(圆心x, 圆心y, 半径):")
    for i, circle in enumerate(circles, 1):
        print(f"圆形{i}: {circle}")
    plot_shapes(circles, MAP_X_RANGE, MAP_Y_RANGE)
    
    print("\n=== 示例2: 生成3个多边形 ===")
    polygons = generate_shapes(
        types='polytope',
        num_shapes=3,
        x_range=MAP_X_RANGE,
        y_range=MAP_Y_RANGE,
        vertex_range=(3, 10)  # 3到6个顶点
    )
    print("生成的多边形顶点坐标:")
    for i, poly in enumerate(polygons, 1):
        print(f"多边形{i}: {poly}")
    plot_shapes(polygons, MAP_X_RANGE, MAP_Y_RANGE)