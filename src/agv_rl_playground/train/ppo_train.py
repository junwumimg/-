import time
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import pygame
from agv_rl_playground.envs.grid_world import GridWorldEnv
from agv_rl_playground.agents.ppo_agent import PPOAgent

# ==========================================
# 🔥 Matplotlib 中文汉化配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# ==========================================


def _shortest_path(env):
    """沿 BFS 距离下降方向还原一条从起点到终点的最短路径"""
    path = [env.start_pos]
    r, c = env.start_pos
    while (r, c) != env.goal_pos:
        d = env.dist_to_goal[r, c]
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (0 <= nr < env.rows and 0 <= nc < env.cols
                    and (nr, nc) not in env.obstacles
                    and env.dist_to_goal[nr, nc] == d - 1):
                r, c = nr, nc
                break
        else:
            raise RuntimeError("找不到到终点的最短路径")
        path.append((r, c))
    return path


def _behavior_clone(agent, env, epochs=2000):
    """用 BFS 最短路专家对 actor 做行为克隆预热"""
    states, targets = [], []
    action_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in env.obstacles or (r, c) == env.goal_pos:
                continue
            d = env.dist_to_goal[r, c]
            if d <= 0:
                continue
            best = None
            for a, (dr, dc) in enumerate(action_delta):
                nr, nc = r + dr, c + dc
                if (0 <= nr < env.rows and 0 <= nc < env.cols
                        and (nr, nc) not in env.obstacles
                        and env.dist_to_goal[nr, nc] == d - 1):
                    best = a
                    break
            if best is None:
                continue
            env.state = (r, c)
            states.append(env._get_obs())
            targets.append(best)

    X = torch.FloatTensor(np.array(states)).to(agent.device)
    y = torch.LongTensor(targets).to(agent.device)
    optimizer = torch.optim.Adam(agent.network.actor.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        loss = loss_fn(agent.network.actor(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = (agent.network.actor(X).argmax(1) == y).float().mean().item()
    env.state = env.start_pos
    print(f"[BC 预热] 样本 {len(states)} 个，专家动作准确率 {acc:.1%}")


def train(episodes=1000, entropy_coef=0.01, lr=1e-4,
          exploration_eps=0.3, exploration_decay=0.995,
          gae_lambda=0.95, curriculum=True, update_timestep=2048,
          mixture_eps=0.1, ppo_shaping=0.0, bc_warmstart=True):
    # 1. 实例化环境
    env = GridWorldEnv(render_mode=None)

    # 2. 实例化 PPO 智能体 (注意：你的新环境返回 5 维状态)
    agent = PPOAgent(
        state_size=env.observation_space.shape[0], # 自动获取维度 (5)
        action_size=env.action_space.n,            # 4
        lr=lr,
        entropy_coef=entropy_coef,
        exploration_eps=exploration_eps,
        exploration_decay=exploration_decay,
        gae_lambda=gae_lambda,
        mixture_eps=mixture_eps,
    )

    if bc_warmstart:
        # 先用 BFS 专家把 actor 预热到接近最优；之后 PPO 只做微调
        _behavior_clone(agent, env)
        curriculum = False

    rewards_history = []

    # PPO 的缓冲区；episode_ends 用于 GAE 在回合边界处切断
    states, actions, log_probs, rewards, next_states, dones, episode_ends = \
        [], [], [], [], [], [], []
    # 收集多少步更新一次网络；小环境用 512 比 2048 更新更及时
    
    print("开始训练 PPO...")
    start_time = time.time()
    time_steps = 0
    
    # 课程学习：从离终点 5 步开始，一步一阶段。
    # 只有当前新起点的近期成功率达标后，才把起点再往后退一格。
    if curriculum:
        path = _shortest_path(env)
        max_d = len(path) - 1
        d_stage = 5
        frontier = deque(maxlen=50)
        agent.exploration_min = 0.1  # 课程期间保留最低探索
        print(f"--- 课程开始: d={d_stage}/{max_d} ---")
    else:
        path = None
        max_d = None
        d_stage = None
        frontier = None

    ep_global = 0
    while ep_global < episodes:
        ep_global += 1
        if curriculum:
            # 一半概率在当前最远起点，一半在已学过的起点中混合
            available = list(range(5, d_stage + 1))
            d_ep = d_stage if random.random() < 0.5 else random.choice(available)
            env.start_pos = path[len(path) - 1 - d_ep]
            max_steps = min(200, 5 * d_ep + 50)
        else:
            d_ep = None
            max_steps = 200

        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            # 1. 选动作
            action, log_prob = agent.choose_action(state)

            # 2. 与环境交互
            if ppo_shaping:
                old_dist = int(env.dist_to_goal[env.state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            if ppo_shaping:
                # 只作用于 PPO 的额外强塑形，不修改共享环境
                new_dist = int(env.dist_to_goal[env.state])
                reward += ppo_shaping * (old_dist - new_dist)
            done = terminated or truncated

            # 3. 存入缓冲区
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(terminated)  # 只存 terminated，不存 truncated

            state = next_state
            total_reward += reward
            step_count += 1
            time_steps += 1
            # 本步是否是回合最后一步：正常结束，或撞上步数上限
            episode_ends.append(done or step_count >= max_steps)

            # 4. 收集够了，学习并清空
            if time_steps % update_timestep == 0:
                agent.learn(states, actions, log_probs, rewards,
                            next_states, dones, episode_ends)
                states, actions, log_probs, rewards, next_states, dones, episode_ends = \
                    [], [], [], [], [], [], []

        rewards_history.append(total_reward)
        agent.decay_exploration()

        if curriculum:
            # 保证课程期间始终有一点探索
            agent.exploration_eps = max(agent.exploration_eps, 0.1)
            if d_ep == d_stage:
                frontier.append(int(done))
                # 近期成功率达标才推进到下一个新起点
                if len(frontier) == frontier.maxlen and np.mean(frontier) >= 0.8:
                    if d_stage < max_d:
                        print(f"--- 课程推进: d={d_stage} -> {d_stage + 1} | "
                              f"成功率 {np.mean(frontier):.0%} ---")
                        d_stage += 1
                        frontier.clear()
                        agent.exploration_eps = 0.3
                    else:
                        frontier.clear()

        # 打印进度
        if ep_global % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            extra = f" | 课程 d={d_stage}" if curriculum else ""
            print(f"回合: {ep_global:4d} | 近50回合平均奖励: {avg_reward:7.3f} | "
                  f"近回合步数: {step_count} | 探索率: {agent.exploration_eps:.3f}{extra}")

    # 恢复真实起点，保证后续测试/策略表使用完整迷宫
    env.start_pos = (0, 0)

    # 处理最后不足 2048 步的尾巴
    if len(states) > 0:
        agent.learn(states, actions, log_probs, rewards,
                    next_states, dones, episode_ends)

    print(f"训练完成！耗时 {time.time() - start_time:.2f} 秒。")
    env.close()

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, alpha=0.4, label="每回合奖励")
    window = 20
    if len(rewards_history) >= window:
        smooth = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_history)), smooth, color='red', label=f"{window}回合平均")
    plt.title("PPO 训练迷宫曲线")
    plt.xlabel("回合")
    plt.ylabel("累计奖励")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    return agent


def test(agent, max_steps=200):
    """带渲染测试 PPO 学到的策略"""
    print("\n开始测试 PPO...")
    env = GridWorldEnv(render_mode="human")

    try:
        for ep in range(3):
            state, _ = env.reset()
            done = False
            steps = 0
            print(f"--- 测试回合 {ep+1} ---")

            while not done and steps < max_steps:
                # PPO 测试时直接取概率最大的动作（确定性策略）
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    action_probs, _ = agent.network(state_tensor)
                    action = torch.argmax(action_probs, dim=1).item()

                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                steps += 1
                
                # 处理 Pygame 关闭事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                        
                time.sleep(0.15)  # 放慢速度看走位

            if terminated:
                print(f"✅ 到达终点！用了 {steps} 步")
            else:
                print(f"❌ 未到达终点（{max_steps} 步内，被截断）")
    finally:
        env.close()


def print_neural_policy_table(agent, env):
    """打印 PPO 的动作概率分布表"""
    print("\n" + "="*80)
    print(" 神经网络策略概率一览表 (格式: [上: %, 下: %, 左: %, 右: %] -> 最优动作)")
    print("="*80)
    
    for r in range(env.rows):
        row_str = ""
        for c in range(env.cols):
            if (r, c) in env.obstacles:
                row_str += " [  ####障碍####  ] "
                continue
            if (r, c) == env.goal_pos:
                row_str += " [  ####终点####  ] "
                continue

            # 临时修改 env.state，获取 5 维观测数据
            env.state = (r, c)
            state = env._get_obs() 
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            
            with torch.no_grad():
                # 获取动作概率
                action_probs, _ = agent.network(state_tensor)
                probs = action_probs.cpu().numpy()[0]
            
            best_action = np.argmax(probs)
            action_names = ['↑', '↓', '←', '→']
            
            # 转换成百分比显示
            p_str = f"[{probs[0]*100:3.0f}%,{probs[1]*100:3.0f}%,{probs[2]*100:3.0f}%,{probs[3]*100:3.0f}%]"
            row_str += f" {p_str}->{action_names[best_action]} "
            
        print(row_str)
    print("="*80 + "\n")


if __name__ == "__main__":
    # 训练
    trained_agent = train()
    
    # 打印策略表
    env_for_print = GridWorldEnv(render_mode=None)
    print_neural_policy_table(trained_agent, env_for_print)
    
    # 测试
    test(trained_agent)