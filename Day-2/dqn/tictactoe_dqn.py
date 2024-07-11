import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random, time
from collections import deque
from dqn_mechanics import TicTacToe, TicTacToeGUI


# Define the neural network for the DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Implement the DQN agent
class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.update_target_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999995
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, available_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state).detach().numpy()[0]
        available_indices = [3 * a[0] + a[1] for a in available_actions]
        q_values = [q_values[i] for i in available_indices]
        return available_actions[np.argmax(q_values)]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * np.amax(
                    self.target_model(next_state).detach().numpy()[0]
                )
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state).detach().numpy()[0]
            target_f[3 * action[0] + action[1]] = target
            target_f = torch.FloatTensor(target_f).unsqueeze(0)
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print(f"Epsilon decayed to {self.epsilon}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()


def train_model(env, agent, episodes, batch_size):
    for e in range(episodes):
        state = env.reset()
        state = env.get_state()
        total_reward = 0

        for time in range(9):  # Max moves in Tic-Tac-Toe
            available_actions = env.get_available_actions()
            action = agent.choose_action(state, available_actions)
            reward = 1 if env.is_blocking_move(action, 1) else 0
            done = env.make_move(1, action)
            next_state = env.get_state()
            reward += (
                1
                if done and env.winner == 1
                else (
                    -1
                    if done and env.winner == -1
                    else 0.5 if done and env.winner == 0 else 0
                )
            )
            # total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print(f"Episode {e + 1}/{episodes}, Epsilon: {agent.epsilon:.2f}")
                break
            agent.replay(batch_size)
        agent.update_epsilon()  # Ensure epsilon is updated after each episode
    return agent


# Train the DQN agent
if __name__ == "__main__":
    timed = time.time()
    env = TicTacToe()
    agent = DQNAgent()

    episodes = 10  # Reduced for testing
    batch_size = 64

    agent = train_model(env, agent, episodes, batch_size)
    agent.save("dqn_tictactoe.pth")

    # Integrate with GUI
    agent.load("dqn_tictactoe.pth")
    gui = TicTacToeGUI(agent, env)
    gui.start()
