import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        # Actor 和 Critic 使用独立网络，避免 critic 的梯度破坏 actor 的特征
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # 动作概率用 Softmax 激活
        action_probs = torch.softmax(self.actor(x), dim=-1)
        # 状态价值就是一个实数
        state_value = self.critic(x)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99,
                 clip_ratio=0.2, entropy_coef=0.01, epochs=10,
                 gae_lambda=0.95, minibatch_size=64,
                 exploration_eps=0.3, exploration_min=0.02,
                 exploration_decay=0.995, mixture_eps=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio # PPO的核心：剪刀限制
        self.entropy_coef = entropy_coef # 鼓励探索的系数
        self.epochs = epochs # 一批数据重复训练几次
        self.gae_lambda = gae_lambda # GAE 的 λ，越小偏差越大、方差越小
        self.minibatch_size = minibatch_size # 每个 minibatch 的样本数
        # ε-greedy 行为策略：防止策略在发现关键动作前就饱和锁死
        self.exploration_eps = exploration_eps
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        # 固定概率下限：π_mix = (1-ε)·π + ε/|A|，防止策略饱和导致 PPO 比率被裁剪
        self.mixture_eps = mixture_eps
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.network(state)
            probs = action_probs.squeeze(0)

        # 给每个动作一个固定的概率下限，避免策略饱和
        mix = self.mixture_eps
        if mix > 0:
            probs = (1.0 - mix) * probs + mix / self.action_size

        dist = Categorical(probs)
        action = int(dist.sample().item())
        log_prob = float(dist.log_prob(
            torch.tensor(action, device=self.device)).item())
        return action, log_prob

    def decay_exploration(self):
        self.exploration_eps = max(
            self.exploration_min,
            self.exploration_eps * self.exploration_decay,
        )

    def learn(self, states, actions, log_probs, rewards, next_states, dones,
              episode_ends=None):
        """
        episode_ends: 每条 transition 是否是所在回合的最后一步（terminated 或 truncated）。
                      用于在 GAE 反向累加时切断回合边界。不传时退化为只按 dones 切断。
        """
        if episode_ends is None:
            episode_ends = dones

        # 1. 把这一波收集到的数据转成 Tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        episode_ends = torch.FloatTensor(episode_ends).to(self.device)

        # 2. 计算 GAE 优势
        with torch.no_grad():
            _, values = self.network(states)
            _, next_values = self.network(next_states)
            values = values.squeeze(-1)
            next_values = next_values.squeeze(-1)
            old_values = values  # 更新前的 V(s)，用于 value clipping

            # TD 误差；terminated 时不 bootstrap（dones=1），truncated 时继续 bootstrap
            deltas = rewards + self.gamma * next_values * (1 - dones) - values

            advantages = torch.zeros_like(deltas)
            last_gae = 0.0
            for t in reversed(range(len(rewards))):
                if episode_ends[t].item() > 0.5:
                    last_gae = 0.0
                last_gae = (deltas[t].item()
                            + self.gamma * self.gae_lambda
                            * (1.0 - episode_ends[t].item()) * last_gae)
                advantages[t] = last_gae

            # λ-return 作为 Critic 的训练目标（注意要在优势归一化之前算）
            returns = advantages + values

        # 优势归一化（降方差，PPO 必做）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. 打乱后按 minibatch 做多次 PPO 更新
        n = len(states)
        indices = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                mb = indices[start:start + self.minibatch_size]
                mb_states = states[mb]
                mb_actions = actions[mb]
                mb_old_log_probs = old_log_probs[mb]
                mb_advantages = advantages[mb]
                mb_returns = returns[mb]

                action_probs, mb_values = self.network(mb_states)
                mb_values = mb_values.squeeze(-1)
                # 与采样时保持一致的概率下限
                if self.mixture_eps > 0:
                    action_probs = ((1.0 - self.mixture_eps) * action_probs
                                    + self.mixture_eps / self.action_size)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO 核心：计算新旧策略的比率并截断
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                                    1.0 + self.clip_ratio) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                values_clipped = old_values[mb] + torch.clamp(
                    mb_values - old_values[mb], -0.2, 0.2)
                critic_loss = nn.MSELoss()(values_clipped, mb_returns)

                # 总损失 = Actor损失 + Critic损失 - 熵奖励
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(1e-5, param_group['lr'] * 0.999)

