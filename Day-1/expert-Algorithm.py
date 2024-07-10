import numpy as np
from TTT import TicTacToe, TicTacToeGUI


class MinimaxAgent:
    def __init__(self, player):
        self.player = player  # 1 for AI, -1 for human

    def choose_action(self, state, available_actions):
        _, move = self.minimax(
            np.array(state).reshape(3, 3), len(available_actions), self.player
        )
        return move

    def minimax(self, state, depth, player):
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
            if (
                np.sum(board.diagonal()) == 3
                or np.sum(np.fliplr(board).diagonal()) == 3
            ):
                return 1
            if (
                np.sum(board.diagonal()) == -3
                or np.sum(np.fliplr(board).diagonal()) == -3
            ):
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
                score, _ = self.minimax(state, depth - 1, -player)
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
                score, _ = self.minimax(state, depth - 1, -player)
                state[x, y] = 0
                if score < best:
                    best = score
                    best_move = (x, y)
            return best, best_move


if __name__ == "__main__":
    # Initialize the TicTacToe environment and the Minimax agent
    env = TicTacToe()
    minimax_agent = MinimaxAgent(player=1)  # AI is player 1

    # Initialize the GUI with the Minimax agent
    gui = TicTacToeGUI(agent=minimax_agent, env=env)

    # Start the game
    gui.start()
