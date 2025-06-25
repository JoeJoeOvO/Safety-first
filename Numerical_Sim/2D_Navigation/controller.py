import numpy as np

class AGVController:
    def __init__(self, target_position, k_rho=0.5, k_alpha=1.0, k_beta=-0.5):
        """
        简单的AGV控制器
        
        参数:
            target_position: 目标位置 (x, y)
            k_rho: 距离增益
            k_alpha: 角度增益
            k_beta: 方向增益
        """
        self.target_x, self.target_y = target_position
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_beta = k_beta
    
    def compute_control(self, x, y, theta):
        """
        计算控制指令
        
        参数:
            x, y: AGV当前位置
            theta: AGV当前朝向(弧度)
            
        返回:
            v: 线速度
            omega: 角速度
        """
        # 计算位置误差
        dx = self.target_x - x
        dy = self.target_y - y
        
        # 转换为极坐标
        rho = np.sqrt(dx**2 + dy**2)  # 距离误差
        alpha = (np.arctan2(dy, dx) - theta + np.pi) % (2*np.pi) - np.pi  # 角度误差
        beta = -theta - alpha  # 方向误差
        
        # 简单的控制律
        v = self.k_rho * rho
        omega = self.k_alpha * alpha + self.k_beta * beta
        
        # 限制速度
        v = max(min(v, 1.0), -1.0)  # 限制线速度在[-1, 1] m/s
        omega = max(min(omega, 1.0), -1.0)  # 限制角速度在[-1, 1] rad/s
        
        return v, omega