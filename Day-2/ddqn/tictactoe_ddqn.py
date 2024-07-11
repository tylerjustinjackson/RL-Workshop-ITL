import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from ddqn_mechanics import TicTacToe, TicTacToeGUI


class DDQNAgent(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        alpha=0.001,
        gamma=0.999,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999999999,
    ):
        super(DDQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size),
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.array(state).flatten()
        if np.random.rand() <= self.epsilon:
            return random.choice(self.get_available_actions(state))
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            act_values = self.model(state).numpy()
        available_actions = self.get_available_actions(state)
        action_index = np.argmax(
            [act_values[action[0] * 3 + action[1]] for action in available_actions]
        )
        return available_actions[action_index]

    def get_available_actions(self, state):
        board = np.array(state).reshape((3, 3))
        return list(zip(*np.where(board == 0)))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).flatten()
            next_state = torch.tensor(next_state, dtype=torch.float32).flatten()
            target = self.model(state).detach().clone()
            action_index = (
                action[0] * 3 + action[1]
            )  # Convert (i, j) action to a single index

            if done:
                target[action_index] = reward
            else:
                t = self.target_model(next_state).detach()
                target[action_index] = reward + self.gamma * torch.max(t)

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def train_agent(agent, env, episodes, batch_size):
    rl_player_wins = 0
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 9])
        while not env.done:
            action = agent.act(state)
            env.make_move(1, action)
            reward = 1 if env.winner == 1 else -2 if env.winner == -1 else 0
            next_state = env.get_state()
            next_state = np.reshape(next_state, [1, 9])
            agent.remember(state, action, reward, next_state, env.done)
            state = next_state
            if env.done:
                agent.update_target_model()
                if env.winner == 1:
                    rl_player_wins += 1
                break
            available_actions = env.get_available_actions()
            random_action = random.choice(available_actions)
            env.make_move(-1, random_action)
            reward = 1 if env.winner == 1 else -2 if env.winner == -1 else 0
            next_state = env.get_state()
            next_state = np.reshape(next_state, [1, 9])
            agent.remember(state, random_action, reward, next_state, env.done)
            state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        print(
            f"Episode {e+1}/{episodes} - Epsilon: {agent.epsilon:.2f} - RL Player Wins: {rl_player_wins}"
        )
    print("Training completed")
    print("Total RL Player Wins during training:", rl_player_wins)


if __name__ == "__main__":
    env = TicTacToe()
    state_size = 9
    action_size = 9
    agent = DDQNAgent(state_size, action_size)

    # Train the agent
    start_time = time.time()
    train_agent(agent, env, episodes=10000000, batch_size=128)

    elapsed_time_seconds = time.time() - start_time
    days = elapsed_time_seconds // (24 * 3600)
    elapsed_time_seconds %= 24 * 3600
    hours = elapsed_time_seconds // 3600
    elapsed_time_seconds %= 3600
    minutes = elapsed_time_seconds // 60
    seconds = elapsed_time_seconds % 60

    print(
        (
            f"Elapsed time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds\n"
        )
    )

    # Save the trained model
    agent.save("ddqn_tictactoe.pth")

    # Load the trained model
    agent.load("ddqn_tictactoe.pth")

    # Start the GUI game
    game = TicTacToeGUI(agent, env)
    game.start()
