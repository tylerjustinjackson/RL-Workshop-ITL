import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk


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


class DDQNAgent(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        alpha=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
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
            self.window.update()  # Update the GUI before AI makes its move
            if not self.env.done:
                self.window.after(
                    500, self.ai_move
                )  # Delay AI move to allow the GUI to update
            else:
                self.window.after(500, self.show_result)  # Delay showing result

    def ai_move(self):
        state = self.env.get_state()
        state = np.reshape(state, [1, 9])
        available_actions = self.agent.get_available_actions(state)
        action = self.agent.act(state)
        self.env.make_move(1, action)
        self.buttons[action[0]][action[1]].config(text="X", state=tk.DISABLED)
        if not self.env.done:
            self.window.update()
        else:
            self.window.after(500, self.show_result)  # Delay showing result

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


if __name__ == "__main__":
    env = TicTacToe()
    state_size = 9
    action_size = 9
    agent = DDQNAgent(state_size, action_size)

    # Train the agent
    start_time = time.time()
    train_agent(agent, env, episodes=100000, batch_size=32)

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
