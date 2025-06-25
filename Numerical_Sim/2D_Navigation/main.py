from controller import AGVController
from kinematics import AGVKinematics
import matplotlib.pyplot as plt
import numpy as np

def simulate_agv(start_pos,target_pos,dt=0.02,total_time=10):
    # 初始化控制器和运动学模型
    controller = AGVController(target_position=target_pos)
    kinematics = AGVKinematics(start_position=(start_pos[0],start_pos[1]), start_theta=start_pos[2])
    
    # 仿真参数
    # dt = 0.02  # 时间步长
    # total_time = 10  # 总仿真时间
    steps = int(total_time / dt)
    
    # 存储轨迹
    trajectory = []
    
    
    # 主仿真循环
    for _ in range(steps):
        # 获取当前位置
        x, y, theta = kinematics.get_state()
        
        # 控制器计算控制指令
        v, omega = controller.compute_control(x, y, theta)
        
        # 更新运动学模型
        kinematics.update(v, omega, dt)
        
        # 记录轨迹
        trajectory.append((x, y))
    
    # 绘制轨迹
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='AGV Path')
    plt.plot(start_pos[0],start_pos[1], 'go', label='Start')
    plt.plot(target_pos[0], target_pos[1], 'ro', label='Target')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('AGV Navigation Simulation')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    start_pos=(0,2,np.pi/4)
    target_pos=(5,4)


    simulate_agv(start_pos,target_pos)