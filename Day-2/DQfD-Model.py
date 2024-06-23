import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# Define the TicTacToe environment
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        return self.board

    def get_available_actions(self):
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, player, position):
        if self.board[position] == 0:
            self.board[position] = player
            if self.check_winner(player):
                self.done = True
                self.winner = player
            elif len(self.get_available_actions()) == 0:
                self.done = True
                self.winner = 0
            return True
        return False

    def check_winner(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == player:
            return True
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == player:
            return True
        return False

    def get_state(self):
        return self.board

    def is_blocking_move(self, action, player):
        temp_board = self.board.copy()
        temp_board[action] = player
        return self.check_winner(player)


# Define the Minimax expert algorithm
def minimax(state, depth, player):
    if player == 1:
        best = [-1, -1, -float("inf")]
    else:
        best = [-1, -1, float("inf")]

    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state):
        x, y = cell[0], cell[1]
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == 1:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value

    return best


def empty_cells(state):
    return list(zip(*np.where(state == 0)))


def game_over(state):
    return check_winner(state, 1) or check_winner(state, -1) or not empty_cells(state)


def check_winner(board, player):
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    if board[0, 0] == board[1, 1] == board[2, 2] == player:
        return True
    if board[0, 2] == board[1, 1] == board[2, 0] == player:
        return True
    return False


def evaluate(state):
    if check_winner(state, 1):
        return 1
    elif check_winner(state, -1):
        return -1
    else:
        return 0


# Define the neural network for Q-learning
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


# Implement the DQfD algorithm
class DQfD:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=0.001,
        batch_size=64,
        buffer_size=10000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.rl_first = []  # To track if RL player went first

        self.model = DQN()
        self.target_model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done, rl_first):
        self.buffer.append((state, action, reward, next_state, done, rl_first))
        self.rl_first.append(rl_first)  # Track if RL player went first

    def replay(self):
        if len(self.buffer) < self.batch_size:
            return

        minibatch = random.sample(self.buffer, self.batch_size)
        for state, action, reward, next_state, done, rl_first in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])

            q_values = self.model(state)
            q_value = q_values[action]

            next_q_values = self.target_model(next_state)
            next_q_value = torch.max(next_q_values)

            expected_q_value = reward + (1 - done) * self.gamma * next_q_value

            loss = self.criterion(q_value, expected_q_value)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def act(self, state):
        if np.random.rand() <= 0.1:
            return random.choice(range(self.action_dim))
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def load_expert_data(self, expert_data):
        for data in expert_data:
            self.remember(*data)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()


# Generate expert data on the fly using Minimax
def generate_expert_move(env, minimax):
    available_actions = env.get_available_actions()
    player = 1 if env.board.sum() == 0 else -1
    action = minimax(env.board, len(available_actions), player)[:2]
    return tuple(action)


# Main function
def main(num_episodes):
    env = TicTacToe()
    dqfd = DQfD(state_dim=9, action_dim=9)

    for e in range(num_episodes):
        state = env.reset()
        done = False

        # Randomly determine who starts first
        rl_starts = random.choice([True, False])
        print(f"Episode {e+1} - RL {'first' if rl_starts else 'second'}")

        while not done:
            if rl_starts:
                action = dqfd.act(state.flatten())
                print(f"RL move:")
                print(env.board)
                env.make_move(-1, (action // 3, action % 3))
                next_state = env.get_state()
                reward = 1 if env.winner == -1 else -1 if env.winner == 1 else 0
                done = env.done
                dqfd.remember(
                    state.flatten(), action, reward, next_state.flatten(), done, True
                )  # Store RL move and flag RL went first
                state = next_state
                if done:
                    break

                action = generate_expert_move(env, minimax)  # Expert move
                print(f"Minimax move:")
                print(env.board)
                env.make_move(1, action)
                next_state = env.get_state()
                reward = 1 if env.winner == 1 else -1 if env.winner == -1 else 0
                done = env.done
                dqfd.remember(
                    state.flatten(),
                    action[0] * 3 + action[1],
                    reward,
                    next_state.flatten(),
                    done,
                    False,
                )  # Store expert move and flag expert went first
                state = next_state

            elif not rl_starts:
                action = generate_expert_move(env, minimax)  # Expert move
                print(f"Minimax move:")
                print(env.board)
                env.make_move(1, action)
                next_state = env.get_state()
                reward = 1 if env.winner == 1 else -1 if env.winner == -1 else 0
                done = env.done
                dqfd.remember(
                    state.flatten(),
                    action[0] * 3 + action[1],
                    reward,
                    next_state.flatten(),
                    done,
                    False,
                )  # Store expert move and flag expert went first
                state = next_state

                action = dqfd.act(state.flatten())
                print(f"RL move:")
                print(env.board)
                env.make_move(-1, (action // 3, action % 3))
                next_state = env.get_state()
                reward = 1 if env.winner == -1 else -1 if env.winner == 1 else 0
                done = env.done
                dqfd.remember(
                    state.flatten(), action, reward, next_state.flatten(), done, True
                )  # Store RL move and flag RL went first
                state = next_state
                if done:
                    break

        # Print final board and winner
        print(f"Final Board:")
        print(env.board)
        print(
            f"Winner is {'Player 1' if env.winner == 1 else 'Player -1' if env.winner == -1 else 'Draw'}"
        )

        dqfd.replay()
        if e % 10 == 0:
            dqfd.update_target_model()


if __name__ == "__main__":
    dqfd = DQfD(state_dim=9, action_dim=9)  # Create an instance of DQfD
    main(2)

    # Save the trained model
    model_path = "dqn_tictactoe_model.pth"
    dqfd.save_model(model_path)
    print(f"Saved DQN model to {model_path}")
