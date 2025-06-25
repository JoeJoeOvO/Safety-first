import numpy as np

class AGVKinematics:
    def __init__(self, start_position=(0, 0), start_theta=0):
        """
        AGV运动学模型(差速驱动)
        
        参数:
            start_position: 初始位置 (x, y)
            initial_theta: 初始朝向(弧度)
        """
        self.x, self.y = start_position
        self.theta = start_theta  # 朝向角度
    
    def update(self, v, omega, dt):
        """
        更新AGV状态
        
        参数:
            v: 线速度
            omega: 角速度
            dt: 时间步长
        """
        # 简单欧拉积分
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        
        # 归一化角度到[-pi, pi]
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
    
    def get_state(self):
        """
        获取当前状态
        
        返回:
            x: x坐标
            y: y坐标
            theta: 朝向角度(弧度)
        """
        return self.x, self.y, self.theta