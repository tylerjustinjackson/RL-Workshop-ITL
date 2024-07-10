# imports
import numpy as np  # we will use this to manipulate the TTT board
from tictactoe_mechanics import TicTacToe, TicTacToeGUI  # imports game mechanics and GUI


class MinimaxAgent:
    def __init__(self, player):
        self.player = (
            player  # 1 for AI, -1 for human, will determine if allowed to use algorithm
        )

    """function to allow algorithm to choose next action in sequence
    we want to find the best move to make essentially"""

    def choose_action(self, state, available_actions):
        _, move = self.minimax(
            np.array(state).reshape(3, 3),
            len(available_actions),
            self.player,
            # _ captures best score while move captures the best move
            # minimax will return a tuple of both
        )
        return move  # returns the best move

    """create minimax function, we want to maximize the minimum score we will get back
    from all possible scenarios"""

    def minimax(self, state, depth, player):

        # helper nested function that will detect if game is over
        def game_over(board):
            # check if any row or column has any absolute sum of 3
            # since human is -1*3 and AI is 1*3
            for i in range(3):
                if abs(np.sum(board[i, :])) == 3 or abs(np.sum(board[:, i])) == 3:
                    return True
            # checks if any diagonals have a abs sum of 3
            if (
                abs(np.sum(board.diagonal())) == 3
                or abs(np.sum(np.fliplr(board).diagonal())) == 3
            ):
                return True
            # if both cases fail, we are not at a win state yet
            return False

        # helper nested function that will assign score to different scenarios
        def evaluate(board):
            # checks if any columns or rows have a sum or 3 or -3
            for i in range(3):
                if np.sum(board[i, :]) == 3 or np.sum(board[:, i]) == 3:
                    return 1
                if np.sum(board[i, :]) == -3 or np.sum(board[:, i]) == -3:
                    return -1
            # checks diagonal sums for 3 or -3
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

            # if now winning condition is met, return 0
            return 0

        # helper nested function will find all empty cells on the board
        def empty_cells(board):
            # returns coordinates of empty cells
            return list(zip(*np.where(board == 0)))

        # if the game is over or maximum depth is reached, it will return None for moves
        if game_over(state) or depth == 0:
            return evaluate(state), None

        """this initializes the best score to negative infinity
        will iterate over all empty cells
        simulates minimax recursively and updates the best score and move found 
        after completion, it will return the overall best score and move
        THIS IS ONLY FOR AI PLAYER"""
        if player == 1:  # AI is the maximizer, used only if AI is playing
            """we use negative infinite as the initial comparison
            -infinity is merely a place holder for the worst possible move we can make"""
            best = -float("inf")
            # best move is not yet defined, so we store it as None
            best_move = None
            #iterates over all empty cells
            for x, y in empty_cells(state):
                """defines our current state, set to player to initiate the 
                'what if we made this move'"""
                state[x, y] = player
                """we are only concerned with getting the best score
                move is stored as dummy variable
                recursive step to find the best score"""
                score, _ = self.minimax(state, depth - 1, -player)
                #once done erase the state
                state[x, y] = 0
                # compare score, update if better than previous best
                if score > best:
                    best = score
                    best_move = (x, y)
            
            # return best score and move
            return best, best_move
        
        else:  # Human is the minimizer
            #postive infinty since we want to stary with highest possible value
            best = float("inf")
            #set best move to none until one is found
            best_move = None
            #iterate of empty cells
            for x, y in empty_cells(state):
                #sets player to initiate the "what if we made this move"
                state[x, y] = player
                #finds best score
                score, _ = self.minimax(state, depth - 1, -player)
                #resets state back to empty cell
                state[x, y] = 0
                #compare scores and updates best move 
                if score < best:
                    best = score
                    best_move = (x, y)
            #returns best score and move
            return best, best_move


if __name__ == "__main__":
    # Initialize the TicTacToe environment and the Minimax agent
    env = TicTacToe()
    minimax_agent = MinimaxAgent(player=1)  # AI is player 1

    # Initialize the GUI with the Minimax agent
    gui = TicTacToeGUI(agent=minimax_agent, env=env)

    # Start the game
    gui.start()
