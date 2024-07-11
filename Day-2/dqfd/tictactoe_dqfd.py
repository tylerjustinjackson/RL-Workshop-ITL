import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import tkinter as tk
from dqfd_mechanics import TicTacToe, TicTacToeGUI


# Define the neural network for Q-learning
# this is pre-built by torch to speed up our process
class DQN(nn.Module):

    # we will need to set a few layer presets
    def __init__(self):
        # calls parent class of DQN from torch
        super(DQN, self).__init__()
        # creates layers with 9 input features and 128 output features.
        self.fc1 = nn.Linear(9, 128)
        # creates layers with 128 input features and 128 output features.
        self.fc2 = nn.Linear(128, 128)
        # creats layers with 128 input features and 9 output features.
        self.fc3 = nn.Linear(128, 9)

    # creates foward propagation for nn
    def forward(self, x):
        # ReLU activation function to the output of the first fully connected layer.
        x = torch.relu(self.fc1(x))
        # ReLU activation function to the output of the second fully connected layer.
        x = torch.relu(self.fc2(x))
        # The third fully connected layer.
        x = self.fc3(x)
        # returns layers back
        return x


# Implement the DQfD algorithm
class DQfD:
    def __init__(
        self,
        # Dimension of the state space.
        state_dim,
        # Dimension of the action space.
        action_dim,
        # Discount factor for future rewards.
        gamma=0.99,
        # Learning rate.
        lr=0.001,
        # Size of the mini-batch for training.
        batch_size=32,
        # Maximum size of the experience replay buffer.
        buffer_size=10000,
    ):
        # set all parameters for DQfD class
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

        # pull DQN from torch defined earlier
        self.model = DQN()
        # create target model as instances of the DQN class.
        self.target_model = DQN()
        # set optimizer call as an Adam optimizer, default from Torch
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # set the loss function as the Mean Squared Error loss function.
        self.criterion = nn.MSELoss()

        # Calls the update_target_model method to synchronize the target model with the current model.
        self.update_target_model()

    # Defines a method to update the target model.
    def update_target_model(self):
        # Copies the weights from the current model to the target model.
        self.target_model.load_state_dict(self.model.state_dict())

    # Defines a method to store experiences in the replay buffer.
    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Defines a method to replay experiences from the buffer and train the model.
    def replay(self):
        # Checks if there are enough experiences in the buffer to sample a mini-batch. If not, it exits the method.
        if len(self.buffer) < self.batch_size:
            return

        # Randomly samples a mini-batch of experiences from the buffer.
        minibatch = random.sample(self.buffer, self.batch_size)
        # Converts the sampled experiences to PyTorch tensors and flattens the state and next state.
        for state, action, reward, next_state, done in minibatch:
            # we use flatten() for consistency with the Torch library
            state = torch.FloatTensor(state).flatten()
            next_state = torch.FloatTensor(next_state).flatten()
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])

            # Computes the Q-values for the current state and selects the Q-value for the given action.
            q_values = self.model(state)
            q_value = q_values[action]

            # Computes the Q-values for the next state using the target model and selects the maximum Q-value.
            next_q_values = self.target_model(next_state)
            next_q_value = torch.max(next_q_values)

            """Calculates the expected Q-value using the reward, the discount factor gamma,
            and the maximum Q-value of the next state."""
            expected_q_value = reward + (1 - done) * self.gamma * next_q_value

            # Computes the loss between the Q-value and the expected Q-value.
            loss = self.criterion(q_value, expected_q_value)

            # Performs backpropagation and updates the model's weights.
            self.optimizer.zero_grad()
            # reverse propagates function defined earlier
            loss.backward()
            self.optimizer.step()

    # Defines a method for the agent to select an action based on the current state.
    def act(self, state):
        # Converts the state to a PyTorch tensor, flattens it, and adds a batch dimension.
        state = torch.FloatTensor(state).flatten().unsqueeze(0)
        # With a probability of 0.1, selects a random action for exploration
        if np.random.rand() <= 0.1:
            return random.choice(range(self.action_dim))
        # Computes the Q-values for the state and returns the action with the highest Q-value.
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    # saves model to given path
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    # loads model from being saved
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()


# Generate expert data on the fly using Minimax
def minimax(state, depth, player):

    # helper function from yesterday to determine is game is over
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

    # helper function from yesterday that allows us to evaluate the board and return rewards
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

    # helper function that checks for empty cells
    def empty_cells(board):
        return list(zip(*np.where(board == 0)))

    # Checks if the game is over or the maximum depth has been reached,
    # and returns the evaluation of the state.
    if game_over(state) or depth == 0:
        return evaluate(state), None

    # performs minimax algorithm, as explained yesterday
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


# function to check for empty cells
def empty_cells(state):
    return list(zip(*np.where(state == 0)))


# game over function
def game_over(state):
    return check_winner(state, 1) or check_winner(state, -1) or not empty_cells(state)


# function to check for winner and return true and False
def check_winner(board, player):
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    if board[0, 0] == board[1, 1] == board[2, 2] == player:
        return True
    if board[0, 2] == board[1, 1] == board[2, 0] == player:
        return True
    return False


# evaluates state of game and returns winner
def evaluate(state):
    if check_winner(state, 1):
        return 1
    elif check_winner(state, -1):
        return -1
    else:
        return 0


# give best expert move to DQfD
def generate_expert_move(env):
    available_actions = env.get_available_actions()
    player = 1
    _, action = minimax(env.board, len(available_actions), player)
    return action


# runs model
def run_model(num_expert_games, num_episodes):
    # loops for all expert games
    for e in range(num_expert_games):
        # prints episode we are on so the prohramming can access progress
        print(e)
        # Makes sure tictactoe is clearn before running
        state = env.reset()
        # while game is not over
        while not env.done:
            # printed episode twice so user knows where the program is at
            print(e, e)
            # #gets minimax move
            action = generate_expert_move(env)
            # updates state and reward from minimax move
            next_state, reward, done = env.board, 0, env.done
            # rewards if game is over
            if done:
                reward = 1 if env.winner == 1 else -1 if env.winner == -1 else 0

            # flattens states for consistency with Torch library
            flattened_state = state.flatten()
            flattened_state_next = next_state.flatten()

            # walk back through the history and get newest information
            agent.remember(
                flattened_state,
                action[0] * 3 + action[1],
                reward,
                flattened_state_next,
                done,
            )
            # set state to be next state
            state = next_state
            env.make_move(1, action)

    # Train the agent with the expert data
    # Reduce the number of episodes for faster execution
    for _ in range(num_episodes):
        state = env.reset()
        while not env.done:
            # This selects an action using the agent's policy.
            action = agent.act(state.flatten())
            # This converts the action index to row and column coordinates.
            row, col = divmod(action, 3)
            # This makes a move in the environment for the AI and continues if the move is invalid.
            if not env.make_move(1, (row, col)):
                continue
            # This gets the next state, reward, and done flag from the environment.
            next_state, reward, done = env.board, 0, env.done
            # This sets the reward based on the winner of the game.
            if done:
                reward = 1 if env.winner == 1 else -1 if env.winner == -1 else 0
            # This stores the experience in the agent's replay buffer.
            agent.remember(state.flatten(), action, reward, next_state.flatten(), done)
            # set state as new state now
            state = next_state
            # replay buffer is called and it walks though propgation as defined earlier
            agent.replay()


# Main function
if __name__ == "__main__":
    env = TicTacToe()
    state_dim = 9
    action_dim = 9

    agent = DQfD(state_dim, action_dim)

    # Generate expert data
    num_expert_games = 10  # Reduce the number of expert games for faster execution
    num_episodes = 100000
    run_model(num_expert_games, num_episodes)

    # Launch the GUI
    gui = TicTacToeGUI(agent, env)
    gui.run()
