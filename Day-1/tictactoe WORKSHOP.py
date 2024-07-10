import tkinter as tk
import numpy as np
import random, pickle
from TTT import TicTacToe, TicTacToeGUI


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        next_max = max(
            [
                self.get_q_value(next_state, a)
                for a in self.get_available_actions(next_state)
            ],
            default=0,
        )
        current_q = self.q_table.get((state, action), 0.0)
        self.q_table[(state, action)] = current_q + self.alpha * (
            reward + self.gamma * next_max - current_q
        )

    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        q_values = [self.get_q_value(state, action) for action in available_actions]
        max_q = max(q_values)
        return available_actions[q_values.index(max_q)]

    def get_available_actions(self, state):
        board = np.array(state).reshape((3, 3))
        return list(zip(*np.where(board == 0)))


def train_agent(agent, env, episodes):

    wins = 0
    for _ in range(episodes):
        state = env.reset()
        state = env.get_state()
        episode = []
        while not env.done:
            available_actions = env.get_available_actions()
            action = agent.choose_action(state, available_actions)
            env.make_move(1, action)
            next_state = env.get_state()
            episode.append((state, action, 1))
            state = next_state
            if env.done:
                break
            available_actions = env.get_available_actions()
            action = random.choice(available_actions)
            env.make_move(-1, action)
            state = env.get_state()
            episode.append((state, action, -1))

        reward = 1 if env.winner == 1 else -2 if env.winner == -1 else 0
        for state, action, player in reversed(episode):
            if player == -1 and env.is_blocking_move(action, 1):
                reward = -1
                # Small negative reward for blocking opponent's winning move
            agent.update_q_value(
                state, action, reward if player == 1 else -reward, state
            )
            reward = reward * agent.gamma

        if env.winner == 1:
            wins += 1

    print(f"win rate: {(wins/episodes)*100}%")


def save_q_table(agent, filename):
    with open(filename, "wb") as f:
        pickle.dump(agent.q_table, f)


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
