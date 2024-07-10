# imports
import numpy as np
import random, pickle
from TTT import TicTacToe, TicTacToeGUI


# defined a class for agent that will learn to play TTT
class QLearningAgent:
    # init accepts different parameters
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):

        # create class variables, q table will be dictionary to store all possible states
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    # retrieves q value for a given state and action from the q table
    def get_q_value(self, state, action):
        # returns q value from table if it exists, if it does not exist will return 0
        return self.q_table.get((state, action), 0.0)

    # updates the Q-value for a state-action pair
    def update_q_value(self, state, action, reward, next_state):
        # finds the maximum Q-value for the next state across all possible actions
        next_max = max(
            [
                self.get_q_value(next_state, a)
                # iterates over all actions a from state
                for a in self.get_available_actions(next_state)
            ],
            default=0,  # if no action from state return zero
        )
        # gets current q value from each state action pair
        current_q = self.q_table.get((state, action), 0.0)
        # updates the q table using the q function update rule
        self.q_table[(state, action)] = current_q + self.alpha * (
            reward + self.gamma * next_max - current_q
        )

    # function to choose action based on state and available actions
    def choose_action(self, state, available_actions):
        # picks number between 0 and 1, function returns a random float
        if random.random() < self.epsilon:
            # if under epsilon, pick a random choice (explore)
            return random.choice(available_actions)
        # if over epsilon, find the best move to make (exploitation)
        q_values = [self.get_q_value(state, action) for action in available_actions]
        # get the max q value out of list of values
        max_q = max(q_values)
        # return best move by finding location of best value
        return available_actions[q_values.index(max_q)]

    # function to get all available actions on board
    def get_available_actions(self, state):
        # reshape board to for consistency with TTT package
        board = np.array(state).reshape((3, 3))
        # return list of coordinates where empty cells are
        return list(zip(*np.where(board == 0)))


# function will train q learning agent
def train_agent(agent, env, episodes):
    """wins start at zero, this will be used to track success of model
    50% + means it is working"""
    wins = 0
    # play number of episodes specificed in intitation
    for _ in range(episodes):
        # reset environment
        state = env.reset()
        # get initial state
        state = env.get_state()
        # create empty list for episodes to track history
        episode = []
        # while environment is still running and we are not out of episodes
        while not env.done:
            # get availible actions
            available_actions = env.get_available_actions()
            # tell agent to make an action given state and available actions
            action = agent.choose_action(state, available_actions)
            # make a move in the environment for AI player
            env.make_move(1, action)
            # get current state from move just made
            next_state = env.get_state()
            # add moves to episdoe history
            episode.append((state, action, 1))
            # set state to whatever next state was from earlier
            state = next_state
            # if out of episodes, break loop
            if env.done:
                break
            # get availible actions for other player
            available_actions = env.get_available_actions()
            """play against random player so we train in all possible scenarios
            if we used a smart RL model we might not get to all states"""
            action = random.choice(available_actions)
            env.make_move(-1, action)
            state = env.get_state()
            # append to episode history
            episode.append((state, action, -1))

        # distribute rewards, 1 for RL, -1 is random player wins
        reward = 1 if env.winner == 1 else -1 if env.winner == -1 else 0
        # back track over history to not give other player chance to block your win
        for state, action, player in reversed(episode):
            # detect if player was blocked over history
            if player == -1 and env.is_blocking_move(action, 1):
                reward = -1
                # Small negative reward for blocking opponent's winning move
            agent.update_q_value(
                # if positive reward positive 1, if not reward negative 1
                state,
                action,
                reward if player == 1 else -reward,
                state,
            )
            reward = reward * agent.gamma
        # tracks wins for all episodes to test effectiveness
        if env.winner == 1:
            wins += 1
    # print final win rate
    print(f"win rate: {(wins/episodes)*100}%")


# save q table to physical file
def save_q_table(agent, filename):
    with open(filename, "wb") as f:
        pickle.dump(agent.q_table, f)


# load q table from file
def load_q_table(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    env = TicTacToe()
    agent = QLearningAgent()

    # Uncomment these lines if you need to train and save the Q-table
    # train_agent(agent, env, 1000000)
    save_q_table(agent, "q_table.pkl")

    # Load the Q-table for playing
    agent.q_table = load_q_table("q_table.pkl")

    # Start the GUI game
    game = TicTacToeGUI(agent, env)
    game.start()

    # After the game ends, print the number of times RL player wins
    # print("RL Player Wins:", env.rl_player_wins)
