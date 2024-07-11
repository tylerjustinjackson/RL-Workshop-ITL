# imports
import numpy as np
import random, torch
import torch.nn as nn
import torch.optim as optim
from ddqn_mechanics import TicTacToe, TicTacToeGUI


class DDQNAgent(nn.Module):
    def __init__(
        self,
        # size of the state space
        state_size,
        # size of the action space
        action_size,
        # learning rate
        alpha=0.001,
        # discount factor for future rewards
        gamma=0.999,
        # exploration rate
        epsilon=1.0,
        # minimum exploration rate
        epsilon_min=0.01,
        # rate at which exploration rate decays
        epsilon_decay=0.9999995,
    ):
        # Calls the parent class (nn.Module) initializer.
        super(DDQNAgent, self).__init__()
        # Sets the class variables with the initialized parameters.
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        # Initializes the main Q-network by calling the _build_model method.
        self.model = self._build_model()
        # Initializes the target Q-network by calling the _build_model method.
        self.target_model = self._build_model()
        # Synchronizes the target model with the main model.
        self.update_target_model()
        # Sets up the Adam optimizer for the main model with the specified learning rate.
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    # Defines a method to build the neural network model.
    def _build_model(self):
        # Creates a sequential neural network
        model = nn.Sequential(
            # An input layer with state_size inputs and 24 outputs
            nn.Linear(self.state_size, 24),
            # A ReLU activation function
            nn.ReLU(),
            # Another ReLU activation function
            nn.Linear(24, 24),
            # An output layer with 24 inputs and action_size outputs
            nn.ReLU(),
            # Returns the constructed model
            nn.Linear(24, self.action_size),
        )
        # returns our model back
        return model

    # Defines a method to update the target model by copying the weights from the main model.
    def update_target_model(self):
        # Copies the weights from the main model to the target model.
        self.target_model.load_state_dict(self.model.state_dict())

    # Defines a method to store experiences in the replay memory.
    def remember(self, state, action, reward, next_state, done):
        # Appends the experience tuple (state, action, reward, next_state, done) to the memory.
        self.memory.append((state, action, reward, next_state, done))

    # Defines a method to select an action based on the current state.
    def act(self, state):
        # Flattens the state array for consistency with Torch
        state = np.array(state).flatten()
        # With probability epsilon, selects a random action, for exploration
        if np.random.rand() <= self.epsilon:
            return random.choice(self.get_available_actions(state))
        # Otherwise, converts the state to a PyTorch tensor, passes it through the main model to
        # get Q-values, and converts the output to a NumPy array.
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            act_values = self.model(state).numpy()

        # Gets the list of available actions for the current state.
        available_actions = self.get_available_actions(state)
        # Finds the index of the action with the highest Q-value among the available actions.
        action_index = np.argmax(
            [act_values[action[0] * 3 + action[1]] for action in available_actions]
        )
        # Returns the action corresponding to the highest Q-value.
        return available_actions[action_index]

    # Defines a method to get the available actions for the current state.
    def get_available_actions(self, state):
        # Reshapes the state array into a 3x3 board.
        board = np.array(state).reshape((3, 3))
        # Returns a list of coordinates where the board is empty (available actions).
        return list(zip(*np.where(board == 0)))

    # Defines a method to train the model by replaying experiences from the memory.
    def replay(self, batch_size):
        # Samples a random mini-batch of experiences from the memory.
        minibatch = random.sample(self.memory, batch_size)
        # Iterates over the experiences in the mini-batch.
        for state, action, reward, next_state, done in minibatch:
            # Converts the state and next state to PyTorch tensors and flattens them.
            state = torch.tensor(state, dtype=torch.float32).flatten()
            next_state = torch.tensor(next_state, dtype=torch.float32).flatten()
            # Gets the Q-values for the current state from the main model and detaches them from the computation graph.
            target = self.model(state).detach().clone()
            # Converts the (i, j) action into a single index.
            action_index = action[0] * 3 + action[1]
            # If the episode is done, sets the target Q-value for the taken action to the reward.
            if done:
                target[action_index] = reward
            # Otherwise, gets the Q-values for the next state from the target model and sets the target Q-value for the taken
            # action to the reward plus the discounted maximum Q-value for the next state.
            else:
                t = self.target_model(next_state).detach()
                target[action_index] = reward + self.gamma * torch.max(t)

            # Resets the gradients of the optimizer.
            self.optimizer.zero_grad()
            # Computes the MSE loss between the predicted Q-values and the target Q-values.
            loss = nn.MSELoss()(self.model(state), target)
            # Performs backpropagation to compute the gradients.
            loss.backward()
            # Updates the model's weights using the computed gradients.
            self.optimizer.step()

        # Decays the exploration rate (epsilon) if it is above the minimum value.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Defines a method to load the model's weights from a file.
    def load(self, name):
        # Loads the model's weights from the specified file.
        self.model.load_state_dict(torch.load(name))

    # Defines a method to save the model's weights to a file.
    def save(self, name):
        # Saves the model's weights to the specified file.
        torch.save(self.model.state_dict(), name)


# Defines a function to train the DDQN agent.
def train_agent(agent, env, episodes, batch_size):

    # Initializes a counter for the number of wins by the RL player.
    rl_player_wins = 0
    # Iterates over the specified number of episodes.
    for e in range(episodes):
        # Resets the environment and reshapes the initial state.
        state = env.reset()
        state = np.reshape(state, [1, 9])
        # Loops until the episode is done.
        while not env.done:
            # Selects an action using the agent.
            action = agent.act(state)
            # Makes the selected move in the environment.
            env.make_move(1, action)
            # Calculates the reward based on the game outcome.
            reward = 1 if env.winner == 1 else -2 if env.winner == -1 else 0
            # Gets the next state and reshapes it.
            next_state = env.get_state()
            next_state = np.reshape(next_state, [1, 9])
            # Stores the experience in the agent's memory.
            agent.remember(state, action, reward, next_state, env.done)
            # update state to next state
            state = next_state
            # If the episode is done, updates the target model and increments the win counter if the RL player won.
            if env.done:
                agent.update_target_model()
                if env.winner == 1:
                    rl_player_wins += 1
                break
            # Selects a random action for the opponent, makes the move, calculates the reward, gets the next state, stores
            # the experience, and updates the current state.
            available_actions = env.get_available_actions()
            random_action = random.choice(available_actions)
            env.make_move(-1, random_action)
            reward = 1 if env.winner == 1 else -2 if env.winner == -1 else 0
            next_state = env.get_state()
            next_state = np.reshape(next_state, [1, 9])
            agent.remember(state, random_action, reward, next_state, env.done)
            # set state to next state
            state = next_state
        # If there are enough experiences in memory, trains the agent by replaying experiences.
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # Prints the episode number, exploration rate, and number of wins by the RL player.
        print(
            f"Episode {e+1}/{episodes} - Epsilon: {agent.epsilon:.2f} - RL Player Wins: {rl_player_wins}"
        )
    # Prints a message indicating that training is completed and the total number of wins by the RL player.
    print("Training completed")
    print("Total RL Player Wins during training:", rl_player_wins)


# Defines a function to check if CUDA (GPU) is available and sets the device accordingly.
def use_gpu():
    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    env = TicTacToe()
    state_size = 9
    action_size = 9
    agent = DDQNAgent(state_size, action_size)

    # Train the agent
    train_agent(agent, env, episodes=100000, batch_size=16)

    # Save the trained model
    agent.save("ddqn_tictactoe.pth")

    # Load the trained model
    agent.load("ddqn_tictactoe.pth")

    # Start the GUI game
    game = TicTacToeGUI(agent, env)
    game.start()
