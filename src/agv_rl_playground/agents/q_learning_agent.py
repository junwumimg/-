import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, 
                 learning_rate=0.1, 
                 gamma=0.95, 
                 epsilon=1.0, 
                 epsilon_decay=0.995, 
                 epsilon_min=0.01):
        
        """
        初始化 Q-learning 智能体
        """

        self.state_size = state_size
        self.action_size = action_size

        # 超参数
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 核心：Q 表。初始化为全 0
        # 形状为 (状态数, 动作数)
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        """
        根据当前状态选择动作（使用 epsilon-greedy 策略）
        """
        # 1. 探索（Explore）：以 epsilon 的概率随机选
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # 2. 利用（Exploit）：选当前 Q 值最大的动作
        return int(np.argmax(self.q_table[state]))
    
    def learn(self, state, action, reward, next_state, done):
        """
        核心学习过程：根据环境反馈更新 Q 表
        """
        # 1. 当前状态的估计值 Q(S, A)
        current_q = self.q_table[state, action]
        
        # 2. 计算目标值 Target
        if done:
            # 如果到达终点/游戏结束，没有未来了，目标就是当前的奖励
            target_q = reward
        else:
            # 如果还没结束，目标 = 当前奖励 + 折扣后的下一状态最大 Q 值
            # 注意：这里用的是 max，也就是那个“想象中的最优解”
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
            
        # 3. 计算 TD 误差，并更新 Q 表
        td_error = target_q - current_q
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        """
        回合结束后调用：让智能体慢慢收起好奇心，越来越专注
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)











