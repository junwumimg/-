import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from collections import deque
class GridWorldEnv(gym.Env):
    """
    一个最基础的 4x4 网格迷宫环境
    符合 Gymnasium 标准接口
    """
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        
        # 1. 动作空间：4个离散动作
        self.action_space = spaces.Discrete(4)
        
        # 2. 观测空间：当前坐标 (x, y) + 相对终点坐标 (dx, dy) + 到终点的 BFS 距离
        #    与 _get_obs() 的返回值保持一致的 5 维
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
       
        # 地图设置
        self.rows = 10
        self.cols = 20
        self.counter = np.zeros(self.rows * self.cols)  # 全局访问统计，仅用于调试
        # 起点和终点
        self.start_pos = (0, 0)
        self.goal_pos = (9, 19)
        # 生成回廊式迷宫墙壁
        self.obstacles = []
        
        # 第一道墙（留右侧缺口）
        for c in range(0, 15):
            self.obstacles.append((2, c))
            
        # 第二道墙（留左侧缺口）
        for c in range(5, 20):
            self.obstacles.append((5, c))
            
        # 第三道墙（留右侧缺口）
        for c in range(0, 15):
            self.obstacles.append((7, c))
            
        # 墙壁内部再随机加几个“柱子”，增加迷惑性
        self.obstacles.extend([(1, 5), (3, 10), (4, 15), (8, 12)])
        
        # ⚠️ 性能优化：转为 set，查找速度从 O(N) 变成 O(1)
        self.obstacles = set(self.obstacles)
        # 预计算每个可走格子到终点的真实最短路距离（BFS），用于奖励塑形和观测
        self.dist_to_goal = self._build_distance_map()
        self.max_dist = max(1, int(self.dist_to_goal.max()))
        self.state = self.start_pos


        # 初始化 Pygame
        self.window = None
        self.clock = None
        self.cell_size = 60


    def _build_distance_map(self):
        """BFS 计算每个可走格子到终点的最短步数，不可达为 -1"""
        dist = -np.ones((self.rows, self.cols), dtype=np.int32)
        gr, gc = self.goal_pos
        dist[gr, gc] = 0
        queue = deque([self.goal_pos])
        while queue:
            r, c = queue.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and (nr, nc) not in self.obstacles
                        and dist[nr, nc] == -1):
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
        return dist

    def _get_obs(self):
        # 当前坐标归一化
        x = self.state[0] / self.rows
        y = self.state[1] / self.cols
        # 目标相对坐标
        dx = (self.goal_pos[0] - self.state[0]) / self.rows
        dy = (self.goal_pos[1] - self.state[1]) / self.cols
        # BFS 最短路距离（归一化），让网络直接知道当前在最短路上的进度
        dist = max(0, int(self.dist_to_goal[self.state[0], self.state[1]]))
        dist_norm = dist / self.max_dist

        return np.array([x, y, dx, dy, dist_norm], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_pos
        return self._get_obs(), {}

    def step(self, action):
        r, c = self.state
        # 移动前的真实最短路距离，用于距离塑形
        old_dist = int(self.dist_to_goal[r, c])

        # 计算移动后的位置
        if action == 0:   nr, nc = r - 1, c
        elif action == 1: nr, nc = r + 1, c
        elif action == 2: nr, nc = r, c - 1
        elif action == 3: nr, nc = r, c + 1

        # 边界和障碍物判断
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            nr, nc = r, c  # 撞墙留在原地
            reward = -1.0
        elif (nr, nc) in self.obstacles:
            nr, nc = r, c 
            reward = -3.0
        else:
            # ==========================================
            # 🔥 距离启发奖励：使用 BFS 真实最短路距离
            current_dist = int(self.dist_to_goal[nr, nc])
            # 归一化距离 (0 到 1 之间)
            normalized_dist = current_dist / (self.rows * self.cols)
            
            # 基础惩罚 -0.1，距离越远扣分越多
            reward = -0.1 - (normalized_dist * 0.1)
            # ==========================================

        # 全局访问统计，仅用于调试，不进入 reward
        self.counter[nr * self.cols + nc] += 1

        # 距离塑形：沿真实最短路靠近终点给正奖励，远离给负惩罚
        # 系数 0.5 明显强于每步 -0.1 的基础惩罚，保证“沿最短路前进”有清晰的正信号
        # 撞墙时位置没变，old_dist == new_dist，塑形项自动为 0
        new_dist = int(self.dist_to_goal[nr, nc])
        reward += 0.5 * (old_dist - new_dist)

        self.state = (nr, nc)

        # 判断是否到达终点
        terminated = (self.state == self.goal_pos)
        if terminated:
            reward = 1.0  # 到达终点给大奖励

        truncated = False
        info = {}

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.cols * self.cell_size, self.rows * self.cell_size)
            )
            pygame.display.set_caption("Grid World")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.window.fill((255, 255, 255))

        for (r, c) in self.obstacles:
            pygame.draw.rect(self.window, (50, 50, 50), 
                             (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

        gr, gc = self.goal_pos
        pygame.draw.rect(self.window, (0, 255, 0), 
                         (gc * self.cell_size, gr * self.cell_size, self.cell_size, self.cell_size))

        r, c = self.state
        pygame.draw.circle(self.window, (255, 0, 0), 
                           (int((c + 0.5) * self.cell_size), int((r + 0.5) * self.cell_size)), 
                           self.cell_size // 3)

        for i in range(self.rows + 1):
            pygame.draw.line(self.window, (200, 200, 200), (0, i * self.cell_size), (self.cols * self.cell_size, i * self.cell_size))
        for j in range(self.cols + 1):
            pygame.draw.line(self.window, (200, 200, 200), (j * self.cell_size, 0), (j * self.cell_size, self.rows * self.cell_size))

        pygame.display.flip()
        self.clock.tick(10)
        
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
