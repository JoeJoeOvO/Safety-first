import numpy as np
from cvxopt import matrix, solvers

class AGVController:
    def __init__(self, target_position, k_rho=0.5, k_alpha=1.0, k_beta=-0.5,min_input=[0,-1],max_input=[5,1]):
        """
        简单的AGV控制器s
        
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
        self.min_input=min_input
        self.max_input=max_input
    
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
    
    def clf(self, x, y, theta,target=None,k_clf=1):
        """
        控制Lyapunov函数(CLF)
        包含目标距离和方向对齐两个部分
        
        参数:
            x, y: 当前位置
            theta: 当前朝向角度
            
        返回:
            v_clf: Lyapunov函数值
            target_dis: 目标距离平方
            similar_theta: 方向余弦相似度
        """
     
        if target is not None:
            self.target_x, self.target_y = target  # 假设target是(x,y)元组        # 计算目标距离平方

        target_dis = (self.target_x - x)**2 + (self.target_y - y)**2
        
        # 计算目标方向向量
        dx = self.target_x - x
        dy = self.target_y - y
        
        # 计算当前朝向向量
        current_dir = np.array([np.cos(theta), np.sin(theta)])
        
        # 计算目标方向向量
        if target_dis < 1e-6:  # 避免除以零
            target_dir = np.array([1.0, 0.0])  # 默认方向
        else:
            target_dir = np.array([dx, dy]) / np.sqrt(target_dis)
        
        # 计算方向余弦相似度 (点积)
        similar_theta = np.dot(current_dir, target_dir)
        
        # CLF值是距离项和方向项的组合
        # 可以根据需要调整权重
        v_clf = target_dis +k_clf*(1 - similar_theta)  # 1-cos(theta)表示方向偏差
        
        # 计算梯度
        if target_dis < 1e-6:
            # 在目标点附近时的特殊处理
            grad_v = np.zeros(3)
        else:
            # 距离项的梯度
            grad_dist_x = -2 * dx
            grad_dist_y = -2 * dy
            grad_dist_theta = 0.0
            
            # 方向项的梯度
            denom = target_dis ** 1.5  # (dx^2 + dy^2)^(3/2)
            grad_dir_x = (np.cos(theta) * dy**2 - np.sin(theta) * dx * dy) / denom
            grad_dir_y = (np.sin(theta) * dx**2 - np.cos(theta) * dx * dy) / denom
            grad_dir_theta = k_clf*(-dx * np.sin(theta) + dy * np.cos(theta)) / np.sqrt(target_dis)
            
            # 总梯度
            grad_v = np.array([
                grad_dist_x + grad_dir_x,
                grad_dist_y + grad_dir_y,
                grad_dist_theta - grad_dir_theta  # 注意负号因为V_dir = 1 - cos(alpha)
            ])


        return v_clf,grad_v

    def cbf(self,x,y,theta,obstacles,k_cbf=1):
        """
        计算所有障碍物的CBF值和梯度
        
        参数:
            x, y, theta: 机器人状态
            obstacles: 障碍物列表，每个障碍物格式为 [x_obs, y_obs, radius]
            
        返回:
            h_list: 各障碍物的CBF值列表
            grad_h_list: 各障碍物的CBF梯度列表 [dh/dx, dh/dy, dh/dtheta]
        """
        h_list = []
        grad_h_list = []
        
        for obs in obstacles:
            x_obs, y_obs, radius = obs
            dx = x - x_obs
            dy = y - y_obs
            distance_sq = dx**2 + dy**2
            
            # 计算方向余弦相似度
            robot_dir = np.array([np.cos(theta), np.sin(theta)])
            if distance_sq < 1e-6:  # 避免除以零
                obs_dir = np.array([1.0, 0.0])  # 默认方向
                cos_sim = 1.0
            else:
                obs_dir = np.array([dx, dy]) / np.sqrt(distance_sq)
                cos_sim = np.dot(robot_dir, obs_dir)
            
            # CBF值
            h = distance_sq - radius**2 - k_cbf * (1 - cos_sim)
            h_list.append(h)
            
            # 计算梯度
            ## 距离项的梯度
            grad_dist_x = 2 * dx
            grad_dist_y = 2 * dy
            grad_dist_theta = 0.0
            
            ## 方向项的梯度
            if distance_sq < 1e-6:
                grad_dir_x = 0.0
                grad_dir_y = 0.0
                grad_dir_theta = 0.0
            else:
                grad_dir_x = self.k_cbf * (np.cos(theta) * dy**2 - np.sin(theta) * dx * dy) / (distance_sq**1.5)
                grad_dir_y = self.k_cbf * (np.sin(theta) * dx**2 - np.cos(theta) * dx * dy) / (distance_sq**1.5)
                grad_dir_theta = self.k_cbf * (-dx * np.sin(theta) + dy * np.cos(theta)) / np.sqrt(distance_sq)
            
            ## 总梯度
            grad_h = np.array([
                grad_dist_x - grad_dir_x,  # dh/dx
                grad_dist_y - grad_dir_y,  # dh/dy
                -grad_dir_theta            # dh/dtheta (注意负号)
            ])
            grad_h_list.append(grad_h)
        
        return h_list, grad_h_list
    
    def constraints(self,model,x,y,theta,types=None,target=None,obstacles=None,k_clf=1,k_cbf=1):
        f_dynamics,g_dynamics=model.dynamics_affine()
        
        v_clf,grad_v=self.clf(x,y,theta,target,k_clf)
        clf_coeff=0.5

        h_cbf,grad_h=self.cbf(x,y,theta,obstacles,k_cbf)
        cbf_coeff=0.5
        # 权重矩阵 (可根据需要调整)
        Q = np.diag([1.0, 1.0])  # 对角权重矩阵
        c = np.zeros(2)  # 线性项
        
        # 转换为 cvxopt 矩阵
        P = matrix(Q)
        q = matrix(c)

        # CLF 约束: grad_v @ g @ u <= -gamma * clf_v - grad_v @ f
        G_clf = grad_v.reshape(1, 3) @ g_dynamics  # (1,2)
        h_clf = np.array(-clf_coeff * v_clf - grad_v @ f_dynamics)  # (1,)

        # 输入约束
        G_input = np.array([
            [1, 0],   # v <= v_max
            [-1, 0],  # -v <= -v_min (即 v >= v_min)
            [0, 1],   # w <= w_max
            [0, -1]   # -w <= -w_min (即 w >= w_min)
        ])
        h_input = np.array([self.max_input[0], -self.min_input[0], self.max_input[1], -self.min_input[1]])
        
        # print(h_clf,h_input)

        # 合并约束
        G = np.vstack([G_clf, G_input])  # (5,2)
        h = np.hstack([h_clf, h_input])  # (5,)
        
        # 转换为 cvxopt 矩阵
        G = matrix(G)
        h = matrix(h)
        return P,q,G,h
    
    def clf_qp(self,x,y,theta,model,target=None,k_clf=1):
        P, q, G, h=self.constraints(model,x,y,theta,types='clf_cbf_qp',target=target,k_clf=k_clf)        
        # 求解 QP
        solvers.options['show_progress'] = False  # 关闭求解过程输出
        solution = solvers.qp(P, q, G, h)
        [v,w]=np.array(solution['x']).flatten()
        return v,w
    
        # if solution['status'] == 'optimal':
        #     u_opt = np.array(solution['x']).flatten()
        #     return u_opt, solution['status']
        # else:
        #     print(f"QP求解失败，状态: {solution['status']}")
        #     # 返回安全值
        #     v_safe = np.clip(0, v_min, v_max)
        #     w_safe = np.clip(0, w_min, w_max)
        #     return np.array([v_safe, w_safe]), solution['status']

    def clf_cbf_qp(self,x,y,theta,obstacles,model,target=None,k_clf=1,k_cbf=1):
        v_clf,grad_v=self.clf(x,y,theta,target,k_clf)
        h_cbf,grad_h=self.cbf(x,y,theta,obstacles,k_cbf)
        f_dynamics,g_dynamics=model.dynamics_affine()

        clf_coeff=0.5
        # def optimal_decay_clf_cbf_qp(self,x,y,theta):

        # def safety_first_clf_cbf_qp(self,x,y,theta):
