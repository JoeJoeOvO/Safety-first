import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional
from matplotlib.patches import Circle, Polygon
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 简体中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def generate_shapes(types: str, 
                   num_shapes: int, 
                   x_range: Tuple[float, float], 
                   y_range: Tuple[float, float],
                   radius_range: Optional[Tuple[float, float]] = None,
                   vertex_range: Optional[Tuple[int, int]] = None,
                   size_limit: Optional[Tuple[float, float]] = None) -> Union[List[Tuple[float, float, float]], List[List[Tuple[float, float]]]]:
    """
    在地图范围内随机生成图形（支持多边形大小限制）
    
    参数:
        types: 图形类型，'circle'或'polytope'
        num_shapes: 要生成的图形数量
        x_range: x轴范围 (xmin, xmax)
        y_range: y轴范围 (ymin, ymax)
        radius_range: 可选，圆的半径范围 (rmin, rmax)
        vertex_range: 可选，多边形的顶点数量范围 (vmin, vmax)
        size_limit: 可选，多边形的大小限制 (最小半径, 最大半径)
        
    返回:
        对于圆形: 返回圆心和半径列表 [(x1, y1, r1), (x2, y2, r2), ...]
        对于多边形: 返回顶点坐标列表 [[(x1, y1), (x2, y2), ...], [...]]
    """
    # 参数验证
    if types not in ['circle', 'polytope']:
        raise ValueError("types参数必须是'circle'或'polytope'")
    
    if types == 'circle' and radius_range is None:
        raise ValueError("生成圆形需要提供radius_range参数")
    
    if types == 'polytope' and vertex_range is None:
        raise ValueError("生成多边形需要提供vertex_range参数")
    
    if size_limit is not None and types == 'polytope':
        if size_limit[0] <= 0 or size_limit[1] <= 0:
            raise ValueError("size_limit值必须为正数")
        if size_limit[0] > size_limit[1]:
            raise ValueError("size_limit最小值不能大于最大值")
    
    xmin, xmax = x_range
    ymin, ymax = y_range
    
    if types == 'circle':
        rmin, rmax = radius_range
        shapes = []
        for _ in range(num_shapes):
            # 确保圆不会超出边界
            x = np.random.uniform(xmin + rmax, xmax - rmax)
            y = np.random.uniform(ymin + rmax, ymax - rmax)
            r = np.random.uniform(rmin, rmax)
            shapes.append((x, y, r))
        return shapes
    
    elif types == 'polytope':
        vmin, vmax = vertex_range
        shapes = []
        
        for _ in range(num_shapes):
            num_vertices = np.random.randint(vmin, vmax + 1)
            valid_polygon = False
            max_attempts = 100  # 防止无限循环
            attempts = 0
            
            while not valid_polygon and attempts < max_attempts:
                attempts += 1
                
                # 生成候选多边形
                vertices = []
                center_x = np.random.uniform(xmin, xmax)
                center_y = np.random.uniform(ymin, ymax)
                
                if size_limit is None:
                    # 无大小限制的情况
                    for _ in range(num_vertices):
                        x = np.random.uniform(xmin, xmax)
                        y = np.random.uniform(ymin, ymax)
                        vertices.append((x, y))
                else:
                    # 有大小限制的情况
                    min_radius, max_radius = size_limit
                    radius = np.random.uniform(min_radius, max_radius)
                    angles = np.sort(np.random.uniform(0, 2*np.pi, num_vertices))
                    
                    for angle in angles:
                        # 添加一些随机性使多边形不完全规则
                        r = radius * np.random.uniform(0.8, 1.2)
                        x = center_x + r * np.cos(angle)
                        y = center_y + r * np.sin(angle)
                        # 确保顶点在边界内
                        x = np.clip(x, xmin, xmax)
                        y = np.clip(y, ymin, ymax)
                        vertices.append((x, y))
                
                # 检查是否构成有效多边形（至少3个不共线点）
                if len(vertices) >= 3:
                    try:
                        hull = ConvexHull(vertices)
                        if len(hull.vertices) >= 3:
                            # 按极角排序顶点
                            center = np.mean(vertices, axis=0)
                            angles = np.arctan2(
                                np.array([v[1] - center[1] for v in vertices]),
                                np.array([v[0] - center[0] for v in vertices])
                            )
                            vertices = [v for _, v in sorted(zip(angles, vertices))]
                            valid_polygon = True
                    except:
                        continue
            
            if valid_polygon:
                shapes.append(vertices)
            else:
                print(f"警告: 无法在{max_attempts}次尝试内生成有效的{num_vertices}边形")
        
        return shapes

def calculate_polygon_size(vertices: List[Tuple[float, float]]) -> float:
    """计算多边形的大小（外接圆半径）"""
    center = np.mean(vertices, axis=0)
    distances = [np.sqrt((v[0]-center[0])**2 + (v[1]-center[1])**2) for v in vertices]
    return np.max(distances)

def plot_shapes(shapes: Union[List[Tuple[float, float, float]], List[List[Tuple[float, float]]]], 
                x_range: Tuple[float, float], 
                y_range: Tuple[float, float]):
    """可视化生成的图形"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if isinstance(shapes[0], tuple):  # 圆形
        for x, y, r in shapes:
            circle = Circle((x, y), r, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, f"r={r:.1f}", ha='center', va='center')
    else:  # 多边形
        for i, vertices in enumerate(shapes):
            polygon = Polygon(vertices, closed=True, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(polygon)
            # 显示多边形大小
            size = calculate_polygon_size(vertices)
            center = np.mean(vertices, axis=0)
            ax.text(center[0], center[1], f"{len(vertices)}边\n{size:.1f}", 
                   ha='center', va='center')
    
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_aspect('equal')
    ax.grid(True)
    plt.title("随机生成的图形")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.show()

# 示例用法
if __name__ == "__main__":
    MAP_X_RANGE = (0, 10)
    MAP_Y_RANGE = (0, 10)
    
    print("=== 示例1: 生成5个圆形 ===")
    circles = generate_shapes(
        types='circle',
        num_shapes=5,
        x_range=MAP_X_RANGE,
        y_range=MAP_Y_RANGE,
        radius_range=(0.2, 2)
    )
    plot_shapes(circles, MAP_X_RANGE, MAP_Y_RANGE)
    print(circles)
    print("\n=== 示例2: 生成3个多边形（无大小限制）===")
    polygons = generate_shapes(
        types='polytope',
        num_shapes=3,
        x_range=MAP_X_RANGE,
        y_range=MAP_Y_RANGE,
        vertex_range=(3, 10))
    plot_shapes(polygons, MAP_X_RANGE, MAP_Y_RANGE)
    print(polygons)

    print("\n=== 示例3: 生成4个多边形（带大小限制10-20）===")
    limited_polygons = generate_shapes(
        types='polytope',
        num_shapes=4,
        x_range=MAP_X_RANGE,
        y_range=MAP_Y_RANGE,
        vertex_range=(3, 10),
        size_limit=(0.2, 2))
    plot_shapes(limited_polygons, MAP_X_RANGE, MAP_Y_RANGE)
    print(limited_polygons)
