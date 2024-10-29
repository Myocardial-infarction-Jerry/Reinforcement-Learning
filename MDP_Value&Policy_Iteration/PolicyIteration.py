import numpy as np

# 迷宫配置
gridSize = 7
gamma = 0.9
rewardStep = -1
terminalReward = 0
actions = ['↑', '↓', '←', '→']  # 上、右、下、左
optimalPolicy = {...}
optimalValues = {...}


def initializeMaze():  # 随机生成起点、终点和障碍物
    src = (np.random.randint(gridSize), np.random.randint(gridSize))
    dst = (np.random.randint(gridSize), np.random.randint(gridSize))
    while dst == src:
        dst = (np.random.randint(gridSize), np.random.randint(gridSize))
    obstacles = set()
    for _ in range(gridSize):  # 随机放置障碍物
        obs = (np.random.randint(gridSize), np.random.randint(gridSize))
        if obs != src and obs != dst:
            obstacles.add(obs)
    return src, dst, obstacles


def initializePolicyAndValues(src, dst, obstacles):  # 初始化策略和价值函数
    policy = {(i, j): actions[0] for i in range(gridSize)
              for j in range(gridSize)}
    values = {(i, j): -np.inf for i in range(gridSize)
              for j in range(gridSize)}
    values[dst] = np.float64(0)
    return policy, values


def getNextState(state, action):
    # 获取动作导致的下一状态
    x, y = state
    if action == '↑':
        return (max(x - 1, 0), y)
    elif action == '↓':
        return (min(x + 1, gridSize - 1), y)
    elif action == '←':
        return (x, max(y - 1, 0))
    elif action == '→':
        return (x, min(y + 1, gridSize - 1))


def policyEvaluation(policy, values, dst, obstacles, theta=1e-4):  # 评估策略
    while True:
        newValues = values.copy()
        delta = np.float64(0)
        for s in values:
            if s == dst or s in obstacles:
                continue
            newValues[s] = rewardStep + gamma * \
                values[getNextState(s, policy[s])]
            delta = max(delta, abs(newValues[s] - values[s]))
        values = newValues
        if delta < theta:
            break
    return values


def policyImprovement(policy, values, dst, obstacles):  # 改进策略
    policyStable = True
    for s in values:
        if s == dst:
            continue
        oldAction = policy[s]
        policy[s] = actions[np.argmax(
            [rewardStep+gamma*values[getNextState(s, a)] for a in actions])]
        if oldAction != policy[s]:
            policyStable = False

    return policyStable, policy


def policyIteration():
    # 主函数执行 Policy Iteration
    src, dst, obstacles = initializeMaze()
    print(f"src: {src}, dst: {dst}, Obstacles: {obstacles}")

    policy, values = initializePolicyAndValues(
        src, dst, obstacles)

    while True:
        values = policyEvaluation(policy, values, dst, obstacles)
        policyStable, policy = policyImprovement(
            policy, values, dst, obstacles)
        if policyStable:
            break

    policyEvaluation(policy, values, dst, obstacles)
    printInfo(policy, values, src, dst, obstacles)
    return policy, values


# 执行算法并打印最优策略和价值函数
def printInfo(policy, values, src, dst, obstacles):
    print("\nOptimal Policy:")
    for i in range(gridSize):
        for j in range(gridSize):
            if (i, j) == dst:
                print(" G ", end="")
            elif (i, j) in obstacles:
                print(" X ", end="")
            else:
                print(f" {policy[(i, j)]} ", end="")
        print()

    print("\nOptimal Values:")
    for i in range(gridSize):
        print(
            " ".join([f"{round(values[(i, j)], 2):^5}" for j in range(gridSize)]))


def main():
    optimalPolicy, optimalValues = policyIteration()


if __name__ == "__main__":
    main()
