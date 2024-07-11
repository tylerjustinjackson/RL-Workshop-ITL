import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import tkinter as tk


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
        return tuple(self.board.flatten())

    def is_blocking_move(self, action, player):
        temp_board = self.board.copy()
        temp_board[action] = player
        return self.check_winner(player)


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
        batch_size=32,
        buffer_size=10000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

        self.model = DQN()
        self.target_model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.buffer) < self.batch_size:
            return

        minibatch = random.sample(self.buffer, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).flatten()
            next_state = torch.FloatTensor(next_state).flatten()
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
        state = torch.FloatTensor(state).flatten().unsqueeze(0)
        if np.random.rand() <= 0.1:
            return random.choice(range(self.action_dim))
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()


# Generate expert data on the fly using Minimax
def minimax(state, depth, player):
    def game_over(board):
        for i in range(3):
            if abs(np.sum(board[i, :])) == 3 or abs(np.sum(board[:, i])) == 3:
                return True
        if (
            abs(np.sum(board.diagonal())) == 3
            or abs(np.sum(np.fliplr(board).diagonal())) == 3
        ):
            return True
        return False

    def evaluate(board):
        for i in range(3):
            if np.sum(board[i, :]) == 3 or np.sum(board[:, i]) == 3:
                return 1
            if np.sum(board[i, :]) == -3 or np.sum(board[:, i]) == -3:
                return -1
        if np.sum(board.diagonal()) == 3 or np.sum(np.fliplr(board).diagonal()) == 3:
            return 1
        if np.sum(board.diagonal()) == -3 or np.sum(np.fliplr(board).diagonal()) == -3:
            return -1
        return 0

    def empty_cells(board):
        return list(zip(*np.where(board == 0)))

    if game_over(state) or depth == 0:
        return evaluate(state), None

    if player == 1:  # AI is the maximizer
        best = -float("inf")
        best_move = None
        for x, y in empty_cells(state):
            state[x, y] = player
            score, _ = minimax(state, depth - 1, -player)
            state[x, y] = 0
            if score > best:
                best = score
                best_move = (x, y)
        return best, best_move
    else:  # Human is the minimizer
        best = float("inf")
        best_move = None
        for x, y in empty_cells(state):
            state[x, y] = player
            score, _ = minimax(state, depth - 1, -player)
            state[x, y] = 0
            if score < best:
                best = score
                best_move = (x, y)
        return best, best_move


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


def generate_expert_move(env):
    available_actions = env.get_available_actions()
    player = 1
    _, action = minimax(env.board, len(available_actions), player)
    return action


# Define the TicTacToe GUI
class TicTacToeGUI:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.window = tk.Tk()
        self.window.title("Tic-Tac-Toe")
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_widgets()
        self.reset_game()

    def create_widgets(self):
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack()

        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0)

        for i in range(3):
            for j in range(3):
                button = tk.Button(
                    self.button_frame,
                    text=" ",
                    font=("normal", 40, "normal"),
                    width=5,
                    height=2,
                    command=lambda i=i, j=j: self.human_move(i, j),
                )
                button.grid(row=i, column=j)
                self.buttons[i][j] = button

        self.result_frame = tk.Frame(self.main_frame)

        self.result_label = tk.Label(
            self.result_frame, text="", font=("normal", 40, "normal")
        )
        self.result_label.pack()

        self.play_again_button = tk.Button(
            self.result_frame,
            text="Play Again",
            font=("normal", 20, "normal"),
            command=self.reset_game,
        )
        self.play_again_button.pack()

    def human_move(self, row, col):
        if not self.env.done and self.env.board[row, col] == 0:
            self.env.make_move(-1, (row, col))
            self.update_buttons()
            if not self.env.done:
                self.ai_move()

    def ai_move(self):
        state = self.env.get_state()
        action = self.agent.act(state)
        row, col = divmod(action, 3)
        if self.env.make_move(1, (row, col)):
            self.update_buttons()

    def update_buttons(self):
        for i in range(3):
            for j in range(3):
                if self.env.board[i, j] == 1:
                    self.buttons[i][j].config(text="X")
                elif self.env.board[i, j] == -1:
                    self.buttons[i][j].config(text="O")
                else:
                    self.buttons[i][j].config(text=" ")

        if self.env.done:
            if self.env.winner == 1:
                self.result_label.config(text="AI wins!")
            elif self.env.winner == -1:
                self.result_label.config(text="Human wins!")
            else:
                self.result_label.config(text="It's a draw!")
            self.result_frame.grid(row=1, column=0)
        else:
            self.result_frame.grid_forget()

    def reset_game(self):
        self.env.reset()
        self.update_buttons()
        self.result_frame.grid_forget()

    def run(self):
        self.window.mainloop()


# Main function
if __name__ == "__main__":
    env = TicTacToe()
    state_dim = 9
    action_dim = 9

    agent = DQfD(state_dim, action_dim)

    # Generate expert data
    num_expert_games = 10  # Reduce the number of expert games for faster execution
    for e in range(num_expert_games):

        print(e)
        state = env.reset()
        while not env.done:
            print(e, e)
            action = generate_expert_move(env)
            next_state, reward, done = env.board, 0, env.done
            if done:
                reward = 1 if env.winner == 1 else -1 if env.winner == -1 else 0

            flattened_state = state.flatten()
            flattened_state_next = next_state.flatten()

            agent.remember(
                flattened_state,
                action[0] * 3 + action[1],
                reward,
                flattened_state_next,
                done,
            )
            state = next_state
            env.make_move(1, action)

    # Train the agent with the expert data
    num_episodes = 10  # Reduce the number of episodes for faster execution
    for _ in range(num_episodes):
        state = env.reset()
        while not env.done:
            action = agent.act(state.flatten())
            row, col = divmod(action, 3)
            if not env.make_move(1, (row, col)):
                continue
            next_state, reward, done = env.board, 0, env.done
            if done:
                reward = 1 if env.winner == 1 else -1 if env.winner == -1 else 0
            agent.remember(state.flatten(), action, reward, next_state.flatten(), done)
            state = next_state
            agent.replay()

    DQfD.save_model("dqfd_model.pth")

    agent = DQfD.load_model("dqfd_model.pth")

    # Launch the GUI
    gui = TicTacToeGUI(agent, env)
    gui.run()
