import time
from agv_rl_playground.envs.grid_world import GridWorldEnv

def main():
    # 创建环境，开启人类渲染模式（会弹出 pygame 窗口）
    env = GridWorldEnv(render_mode="human")
    
    # 跑 3 个回合
    for episode in range(3):
        state, info = env.reset()
        done = False
        print(f"--- 回合 {episode+1} 开始 ---")
        
        while not done:
            # 纯随机策略：在 0,1,2,3 里随便选一个
            action = env.action_space.sample()
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            time.sleep(0.1) # 停顿一下，方便肉眼看清轨迹
            
        print(f"到达终点！(状态: {state})")
        
    env.close()

if __name__ == "__main__":
    main()
    