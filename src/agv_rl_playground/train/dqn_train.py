import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from agv_rl_playground.envs.grid_world import GridWorldEnv
from agv_rl_playground.agents.dqn_agent import DQNAgent

# ==========================================
# 🔥 关键：Matplotlib 中文汉化配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] # 优先使用黑体/微软雅黑
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号（否则奖励是负数时，负号会乱码）
# ==========================================

def train():
    # 1. 实例化环境（训练时不渲染，速度更快）
    env = GridWorldEnv(render_mode=None)
    
    # 2. 实例化 DQN 智能体
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        batch_size=64,
        epsilon_decay=0.996,  # 1500 回合内探索率能衰减到最小值
    )
    
    episodes = 1000       # DQN 比 Q-table 学得快，500 回合足够了
    rewards_history = [] # 记录每个回合的总奖励，用来画图
    
    print("开始训练 DQN...")
    start_time = time.time()
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0


        while not done and step_count < 100: # 限制最大步数，防死循环
            # 1. 选动作
            action = agent.choose_action(state)
            
            # 2. 与环境交互
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. 存入经验池并学习（每次走一步，就学一次）
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            # 目标网络的同步已经在 agent.learn() 内部按全局学习步数自动完成

        # 每个回合结束，衰减一次探索率
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # 打印进度
        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"回合: {ep+1:3d} | 近50回合平均奖励: {avg_reward:6.3f} | 探索率: {agent.epsilon:.3f}")

    print(f"训练完成！耗时 {time.time() - start_time:.2f} 秒。")
    env.close()

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, alpha=0.4, label="每回合奖励")
    # 画滑动平均线
    window = 20
    if len(rewards_history) >= window:
        smooth = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_history)), smooth, color='red', label=f"{window}回合平均")
    plt.title("DQN 训练迷宫曲线")
    plt.xlabel("回合")
    plt.ylabel("累计奖励")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    return agent

def test(agent, max_steps=200):
    """带渲染测试学到的策略"""
    print("\n开始测试...")
    env = GridWorldEnv(render_mode="human")

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # 测试时纯利用，不探索

    try:
        for ep in range(3):
            state, _ = env.reset()
            terminated = False
            steps = 0
            print(f"--- 测试回合 {ep+1} ---")

            while not terminated and steps < max_steps:
                action = agent.choose_action(state)

                next_state, _, terminated, _, _ = env.step(action)
                state = next_state
                steps += 1
                if steps>= max_steps:
                    print(f"未在 {max_steps} 步内到达终点，强制结束。")
                    terminated = False
                    break

                time.sleep(0.3)  # 放慢速度

            if terminated:
                print(f"到达终点！用了 {steps} 步")
            else:
                print(f"未到达终点（{max_steps} 步内）")
    finally:
        agent.epsilon = old_epsilon  # 恢复
        env.close()



def print_neural_q_table(agent, env):
    print("\n" + "="*60)
    print(" 神经网络 Q 值一览表 (格式: [上, 下, 左, 右])")
    print("="*60)
    
    for r in range(env.rows):
        row_str = ""
        for c in range(env.cols):
            if (r, c) in env.obstacles:
                row_str += " [  ####障碍####  ] "
                continue
            if (r, c) == env.goal_pos:
                row_str += " [  ####终点####  ] "
                continue

            # ==========================================
            # 🔥 关键修复：临时修改 env.state，让 env 自己吐出 4 维状态
            env.state = (r, c)
            state = env._get_obs() 
            # ==========================================
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            
            with torch.no_grad():
                q_values = agent.q_network(state_tensor).cpu().numpy()[0]
            
            best_action = np.argmax(q_values)
            action_names = ['↑', '↓', '←', '→']
            
            q_str = f"[{q_values[0]:5.1f}, {q_values[1]:5.1f}, {q_values[2]:5.1f}, {q_values[3]:5.1f}]"
            row_str += f" {q_str}->{action_names[best_action]} "
            
        print(row_str)
    print("="*60 + "\n")


if __name__ == "__main__":

    """
    env = GridWorldEnv(render_mode=None)
    env.render()
    time.sleep(10)
    """
    trained_agent = train()
    env_for_print = GridWorldEnv(render_mode=None)
    print_neural_q_table(trained_agent, env_for_print)
    test(trained_agent)
    