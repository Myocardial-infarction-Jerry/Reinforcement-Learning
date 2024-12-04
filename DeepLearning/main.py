import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import gymnasium as gym
import pygame
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"


class DqnNetwork(nn.Module):  # Define the DQN network
    def __init__(self, stateDim, actionDim):
        super(DqnNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(stateDim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, actionDim)
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=object)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, nextState, done):
        self.buffer[self.position] = (state, action, reward, nextState, done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batchSize):
        indices = np.random.choice(self.size, batchSize, replace=False)
        samples = [self.buffer[i] for i in indices]
        return zip(*samples)  # Unpack into states, actions, etc.

    def __len__(self):
        return self.size


# Initialize environment and parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

env = gym.make('CartPole-v1')
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.n
replayBuffer = ReplayBuffer(10000)

policyNetwork = DqnNetwork(stateDim, actionDim).to(device)
targetNetwork = DqnNetwork(stateDim, actionDim).to(device)
targetNetwork.load_state_dict(policyNetwork.state_dict())
targetNetwork.eval()

optimizer = optim.Adam(policyNetwork.parameters(), lr=1e-3)
lossFunction = nn.MSELoss()

gamma = 0.99
epsilon = 1.0
epsilonDecay = 0.995
epsilonMin = 0.01
batchSize = 64
targetUpdateFrequency = 10
episodes = 1000
rewardList = []

# Create checkpoint directory
checkpointDir = "ckpt"
os.makedirs(checkpointDir, exist_ok=True)


def saveCheckpoint(model, optimizer, episode, filename):
    checkpoint = {
        'modelStateDict': model.state_dict(),
        'optimizerStateDict': optimizer.state_dict(),
        'episode': episode
    }
    torch.save(checkpoint, filename)


def loadCheckpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['modelStateDict'])
    optimizer.load_state_dict(checkpoint['optimizerStateDict'])
    return checkpoint['episode']


def selectAction(state, epsilon):  # Select action with epsilon-greedy policy
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            stateTensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policyNetwork(stateTensor).argmax().item()


def train(epsilon):  # Training and checkpoint saving
    for episode in tqdm(range(episodes), desc="TrainingProgress"):
        state, _ = env.reset()
        state = np.array(state)
        totalReward = 0

        for t in range(500):
            action = selectAction(state, epsilon)
            nextState, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Check if the cart position exceeds bounds
            cartPosition = nextState[0]
            if abs(cartPosition) > env.unwrapped.x_threshold:
                done = True
                reward = -1.0  # Penalize for going out of bounds

            nextState = np.array(nextState)
            replayBuffer.push(state, action, reward, nextState, done)
            state = nextState
            totalReward += reward

            if done:
                break

            if len(replayBuffer) > batchSize:
                states, actions, rewards_, nextStates, dones = replayBuffer.sample(
                    batchSize)

                statesTensor = torch.FloatTensor(np.array(states)).to(device)
                actionsTensor = torch.LongTensor(
                    actions).unsqueeze(1).to(device)
                rewardsTensor = torch.FloatTensor(
                    rewards_).unsqueeze(1).to(device)
                nextStatesTensor = torch.FloatTensor(
                    np.array(nextStates)).to(device)
                donesTensor = torch.FloatTensor(dones).unsqueeze(1).to(device)

                qValues = policyNetwork(statesTensor).gather(1, actionsTensor)
                nextQValues = targetNetwork(
                    nextStatesTensor).max(1, keepdim=True)[0]
                targetQValues = rewardsTensor + gamma * \
                    nextQValues * (1 - donesTensor)

                loss = lossFunction(qValues, targetQValues)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilonDecay, epsilonMin)
        if episode % targetUpdateFrequency == 0:
            targetNetwork.load_state_dict(policyNetwork.state_dict())

        rewardList.append(totalReward)

        # Save checkpoint every 50 episodes
        if episode % 100 == 0:
            saveCheckpoint(policyNetwork, optimizer, episode, os.path.join(
                checkpointDir, f"dqnCheckpointEp{episode}.pth"))

    # Save final model
    saveCheckpoint(policyNetwork, optimizer, episodes,
                   os.path.join(checkpointDir, "dqnFinal.pth"))

    # Plot Training Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewardList, label="Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpointDir, "TrainingRewards.png"))
    plt.close()


def test(checkpointPath):  # Testing using the trained model
    loadCheckpoint(policyNetwork, optimizer, checkpointPath)
    state, _ = env.reset()
    state = np.array(state)
    totalReward = 0

    for t in range(500):
        action = selectAction(state, 0)  # Epsilon = 0 during testing
        nextState, reward, terminated, truncated, _ = env.step(action)
        state = nextState
        totalReward += reward
        if terminated or truncated:
            break

    env.close()
    print("TestingTotalReward:", totalReward)


def show(checkpointPath):  # Show agent using pygame
    loadCheckpoint(policyNetwork, optimizer, checkpointPath)
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state, _ = env.reset()
    state = np.array(state)

    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption('CartPole Agent')

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action = selectAction(state, 0)  # Epsilon = 0 during testing
        nextState, reward, terminated, truncated, _ = env.step(action)
        state = nextState
        done = terminated or truncated

        frame = env.render()
        frame = np.transpose(frame, (1, 0, 2))
        frameSurface = pygame.surfarray.make_surface(frame)
        screen.blit(frameSurface, (0, 0))

        pygame.display.flip()
        clock.tick(30)

    env.close()
    pygame.quit()


# Uncomment one of the following to train or test
# train(epsilon)  # Train and save checkpoints
# test(os.path.join(checkpointDir, "dqnFinal.pth"))  # Test using the final checkpoint
show(os.path.join(checkpointDir, "dqnFinal.pth"))  # Show agent using pygame
