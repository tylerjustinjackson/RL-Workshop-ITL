import tkinter as tk
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        self.rl_player_wins = 0
        return self.board

    def get_available_actions(self):
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, player, position):
        if self.board[position] == 0:
            self.board[position] = player
            if self.check_winner(player):
                self.done = True
                self.winner = player
                if player == 1:
                    self.rl_player_wins += 1
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


class TicTacToeGUI:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.window = tk.Tk()
        self.window.title("Tic-Tac-Toe")
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_widgets()

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

    def reset_game(self):
        self.env.reset()
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text=" ", state=tk.NORMAL)
        self.result_frame.grid_forget()
        self.button_frame.grid(row=0, column=0)

    def human_move(self, i, j):
        if self.env.make_move(-1, (i, j)):
            self.buttons[i][j].config(text="O", state=tk.DISABLED)
            self.window.update()
            if not self.env.done:
                self.window.after(500, self.ai_move)
            else:
                self.window.after(500, self.show_result)

    def ai_move(self):
        state = self.env.get_state()
        available_actions = self.env.get_available_actions()
        action = self.agent.choose_action(state, available_actions)
        if self.env.make_move(1, action):
            self.buttons[action[0]][action[1]].config(text="X", state=tk.DISABLED)
        if not self.env.done:
            self.window.update()
        else:
            self.window.after(500, self.show_result)

    def show_result(self):
        self.button_frame.grid_forget()
        if self.env.winner == 1:
            result_text = "AI wins!"
        elif self.env.winner == -1:
            result_text = "You win!"
        else:
            result_text = "It's a draw!"
        self.result_label.config(text=result_text)
        self.result_frame.grid(row=0, column=0)

    def start(self):
        self.window.mainloop()


import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

# Hyperparameters
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(BUFFER_SIZE)
        self.steps_done = 0

    def choose_action(self, state, available_actions):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float32)
                q_values = self.policy_net(state_tensor)
                q_values = q_values.squeeze().numpy()
                for action in available_actions:
                    q_values[action[0] * 3 + action[1]] = -np.inf
                action_idx = np.argmax(q_values)
                return (action_idx // 3, action_idx % 3)
        else:
            return random.choice(available_actions)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s.unsqueeze(0) for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state])
        action_batch = torch.cat([a for a in batch.action])
        reward_batch = torch.cat([r for r in batch.reward])

        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )
        expected_state_action_values = (
            next_state_values * GAMMA
        ) + reward_batch.float()

        loss = F.smooth_l1_loss(
            state_action_values.squeeze(), expected_state_action_values
        )

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        self.rl_player_wins = 0  # Counter for RL player wins
        return self.board

    def get_available_actions(self):
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, player, position):
        if self.board[position] == 0:
            self.board[position] = player
            if self.check_winner(player):
                self.done = True
                self.winner = player
                if player == 1:
                    self.rl_player_wins += 1  # Increment RL player wins counter
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
        return torch.tensor(self.board.flatten(), dtype=torch.float32)

    def is_blocking_move(self, action, player):
        temp_board = self.board.copy()
        temp_board[action] = player
        return self.check_winner(player)


if __name__ == "__main__":
    agent = DQNAgent()
    env = TicTacToe()

    # Train the agent
    NUM_EPISODES = 1000
    for episode in range(NUM_EPISODES):
        env.reset()
        state = env.get_state()
        while not env.done:
            available_actions = env.get_available_actions()
            action = agent.choose_action(state, available_actions)
            env.make_move(1, action)
            next_state = env.get_state()
            reward = 0
            if env.winner == 1:
                reward = 1
            elif env.winner == -1:
                reward = -1
            elif env.winner == 0:
                reward = 0.5
            agent.memory.push(
                state,
                torch.tensor([[action[0] * 3 + action[1]]], dtype=torch.long),
                next_state,
                torch.tensor([reward], dtype=torch.float32),
            )
            agent.optimize_model()
            state = next_state

        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    torch.save(agent.policy_net.state_dict(), "dqn_tictactoe.pth")

    # Play against the trained model
    print("Training complete. Now you can play against the AI.")
    gui = TicTacToeGUI(agent, env)
    gui.start()
