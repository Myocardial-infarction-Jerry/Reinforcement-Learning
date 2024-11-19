import numpy as np
import random
import matplotlib.pyplot as plt

# 环境参数
ROWS, COLS = 4, 12  # 网格大小
START = (3, 0)      # 起点
GOAL = (3, 11)      # 目标点
CLIFF = [(i, j) for i in range(2, 4) for j in range(1, 11)]  # 悬崖位置

# 动作定义
ACTIONS = ['up', 'down', 'left', 'right']
actionToDelta = {
    'up': (-1, 0), 
    'down': (1, 0), 
    'left': (0, -1), 
    'right': (0, 1)
}

# 超参数
alpha = 0.1   # 学习率
gamma = 0.9   # 折扣因子
epsilon = 0.1 # ε-贪婪策略

def step(state, action):
    """模拟环境的交互逻辑"""
    delta = actionToDelta[action]
    nextState = (state[0] + delta[0], state[1] + delta[1])
    
    # 边界处理
    nextState = (
        max(0, min(ROWS - 1, nextState[0])),
        max(0, min(COLS - 1, nextState[1]))
    )
    
    # 奖励逻辑
    if nextState in CLIFF:
        return START, -100  # 掉入悬崖，重置到起点
    elif nextState == GOAL:
        return nextState, 0  # 到达目标，奖励为0
    else:
        return nextState, -1  # 每一步的默认奖励为-1

def qLearning(episodes=500):
    """Q-learning 算法实现"""
    qTable = { (i, j): {a: 0 for a in ACTIONS} for i in range(ROWS) for j in range(COLS) }
    totalRewards = []  # 累计奖励记录

    for _ in range(episodes):
        state = START
        episodeReward = 0
        while state != GOAL:
            if random.random() < epsilon:
                action = random.choice(ACTIONS)  # 随机探索
            else:
                action = max(qTable[state], key=qTable[state].get)  # 利用最优策略
            
            nextState, reward = step(state, action)
            episodeReward += reward

            maxQNext = max(qTable[nextState].values())
            qTable[state][action] += alpha * (reward + gamma * maxQNext - qTable[state][action])
            
            state = nextState
        
        totalRewards.append(episodeReward)

    return qTable, totalRewards

def sarsa(episodes=500):
    """SARSA 算法实现"""
    qTable = { (i, j): {a: 0 for a in ACTIONS} for i in range(ROWS) for j in range(COLS) }
    totalRewards = []  # 累计奖励记录

    for _ in range(episodes):
        state = START
        episodeReward = 0
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = max(qTable[state], key=qTable[state].get)

        while state != GOAL:
            nextState, reward = step(state, action)
            episodeReward += reward

            if random.random() < epsilon:
                nextAction = random.choice(ACTIONS)
            else:
                nextAction = max(qTable[nextState], key=qTable[nextState].get)

            qTable[state][action] += alpha * (reward + gamma * qTable[nextState][nextAction] - qTable[state][action])

            state, action = nextState, nextAction
        
        totalRewards.append(episodeReward)

    return qTable, totalRewards

# 可视化训练过程
def visualizeTrainingProcess(rewardsQL, rewardsSARSA, episodes, outputFile="training_comparison.png"):
    """将 Q-learning 和 SARSA 的奖励曲线保存为图片"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(episodes), rewardsQL, label='Q-Learning', alpha=0.8)
    plt.plot(range(episodes), rewardsSARSA, label='SARSA', alpha=0.8)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning vs SARSA Training Rewards')
    plt.legend()
    plt.grid()
    plt.savefig(outputFile)
    plt.close()

# 主函数运行 Q-learning 和 SARSA
if __name__ == "__main__":
    episodes = 200

    # Q-learning
    qLearningTable, qLearningRewards = qLearning(episodes)

    # SARSA
    sarsaTable, sarsaRewards = sarsa(episodes)

    # 可视化并保存
    visualizeTrainingProcess(qLearningRewards, sarsaRewards, episodes, outputFile="training_comparison.png")
    print("训练奖励曲线已保存为 'training_comparison.png'")
