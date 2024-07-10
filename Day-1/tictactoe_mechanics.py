import tkinter as tk
import numpy as np
import random, pickle


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
        available_actions = self.env.get_available_actions()
        action = self.agent.choose_action(state, available_actions)
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
