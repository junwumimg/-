import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 1. 定义神经网络结构
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # 输入层 -> 隐藏层 -> 隐藏层 -> 输出层
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
      
        return self.fc3(x) # 输出层不加激活函数，因为 Q 值可以是任意实数
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size,
                 lr=1e-4, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.005,
                 batch_size=64, buffer_capacity=10000,
                 target_update_freq=500):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._learn_steps = 0

        # 确定设备（GPU 或 CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化两个网络
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() # 目标网络只做推理，不训练
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

    
    def choose_action(self, state):
        # Epsilon-greedy 探索
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # 把状态变成张量，喂给神经网络
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())

    def learn(self, state, action, reward, next_state, done):
        # 存入经验池
        self.memory.push(state, action, reward, next_state, done)
        
        # 经验不够，先不学
        if len(self.memory) < self.batch_size:
            return
        
        # 随机抽取一批经验
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 1. 策略网络算出当前状态下，实际动作的 Q 值
        current_q = self.q_network(states).gather(1, actions)
        
        # 2. Double DQN 目标值：
        #    当前网络选动作，目标网络评估这个动作的价值
        with torch.no_grad():
            best_actions = self.q_network(next_states).argmax(1, keepdim=True)
            max_next_q = self.target_network(next_states).gather(1, best_actions)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # 3. 计算损失并更新策略网络
        loss = nn.SmoothL1Loss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. 每学习 target_update_freq 次，同步一次目标网络
        #    注意：计数器必须全局累加，不能放在每个回合都会归零的局部变量里
        self._learn_steps += 1
        if self._learn_steps % self.target_update_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        """同步目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)