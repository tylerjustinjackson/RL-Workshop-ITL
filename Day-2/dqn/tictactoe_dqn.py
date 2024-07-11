import random, time, torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from dqn_mechanics import TicTacToe, TicTacToeGUI


# Defines a neural network class for Deep Q-Network (DQN)
# that inherits from nn.Module.
class DQN(nn.Module):
    # creates layers for us using Torch preset functionality
    def __init__(self):
        super(DQN, self).__init__()
        # structure with with 9 input features and 128 output features.
        self.fc1 = nn.Linear(9, 128)
        # with 128 input features and 128 output features.
        self.fc2 = nn.Linear(128, 128)
        # with 128 input features and 9 output features.
        self.fc3 = nn.Linear(128, 9)

    # forward propagtion function
    def forward(self, x):
        # ReLU activation function to the output of the first fully connected layer.
        x = torch.relu(self.fc1(x))
        # ReLU activation function to the output of the second fully connected layer.
        x = torch.relu(self.fc2(x))
        # The third fully connected layer without activation.
        x = self.fc3(x)
        # returns findings from propagation
        return x


# Implement the DQN agent
class DQNAgent:
    def __init__(self):
        # starts the main and target DQN models.
        self.model = DQN()
        self.target_model = DQN()
        # Synchronization of the target model with the main model.
        self.update_target_model()
        # creates an experience replay buffer (memory) with a maximum length of 2000.
        self.memory = deque(maxlen=2000)
        # discount factor
        self.gamma = 0.95
        # starting epsilon
        self.epsilon = 1
        # where epsilon should be at a minimimum
        self.epsilon_min = 0.01
        # decary rate of epsilon
        self.epsilon_decay = 0.9999995
        # learning rate for model
        self.learning_rate = 0.001
        # An Adam optimizer and Mean Squared Error (MSE) loss function
        # both presets from Torch
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    # Defines a method to update the target model by copying the weights from the main model.
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Defines a method to store experiences (state, action, reward, next_state, done)
    # in the replay buffer.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # finds best action to make
    def choose_action(self, state, available_actions):
        # explore vs exploit rate as mentioned yesterday
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        # gets states from DQN
        state = torch.FloatTensor(state).unsqueeze(0)
        # updates and gets all possible q values
        q_values = self.model(state).detach().numpy()[0]
        # forms them to a list
        available_indices = [3 * a[0] + a[1] for a in available_actions]
        q_values = [q_values[i] for i in available_indices]
        # selects the action with the highest Q-value from the model's output for the given state
        return available_actions[np.argmax(q_values)]

    def replay(self, batch_size):
        # If there are fewer experiences than batch_size, returns.
        if len(self.memory) < batch_size:
            return
        # Samples a random mini-batch of experiences.
        minibatch = random.sample(self.memory, batch_size)
        # Computes the target Q-value.
        for state, action, reward, next_state, done in minibatch:
            # sets target to be wanting the reward
            target = reward
            # while still playing
            if not done:
                # updates next state
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                #  Updates the Q-value for the taken action.
                target = reward + self.gamma * np.amax(
                    self.target_model(next_state).detach().numpy()[0]
                )
            # updates current state
            state = torch.FloatTensor(state).unsqueeze(0)
            # updates target from earlier presets
            target_f = self.model(state).detach().numpy()[0]
            target_f[3 * action[0] + action[1]] = target
            target_f = torch.FloatTensor(target_f).unsqueeze(0)
            # walks through the optimizer function
            self.optimizer.zero_grad()
            # returns the current model given the state we are in
            output = self.model(state)
            # goes through the loss function to determine how good it is
            loss = self.criterion(output, target_f)
            # Performs backpropagation and updates the model's weights.
            loss.backward()
            self.optimizer.step()

    # Defines a method to decay the exploration rate after each episode.
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print(f"Epsilon decayed to {self.epsilon}")

    # save model to phsyical file
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    # load model from file
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()


# function to walk through training process
def train_model(env, agent, episodes, batch_size):
    # loop through all episodes
    for e in range(episodes):
        # set up envrionment and make sure it is reset and ready to play
        state = env.reset()
        state = env.get_state()

        # for all moves in tictactoe
        for time in range(9):  # Max moves in Tic-Tac-Toe
            # get available actions
            available_actions = env.get_available_actions()
            # get action from agent
            action = agent.choose_action(state, available_actions)
            # reward based on if move was blocked, sismilar to yesterday
            reward = 1 if env.is_blocking_move(action, 1) else 0
            # look at environment and tell if we are done
            done = env.make_move(1, action)
            # get next state
            next_state = env.get_state()
            # reward system, AI is 1 and human is -1
            reward += (
                1
                if done and env.winner == 1
                else (
                    -1
                    if done and env.winner == -1
                    else 0.5 if done and env.winner == 0 else 0
                )
            )
            # load history of moves, same as yesterday with updated for our DQN
            agent.remember(state, action, reward, next_state, done)
            # update state to next state
            state = next_state
            # when done, update our target model
            if done:
                agent.update_target_model()
                print(f"Episode {e + 1}/{episodes}")
                break
            # run replay buffer algorithm
            agent.replay(batch_size)
        # nsure epsilon is updated after each episode
        agent.update_epsilon()
    # returns our DQN agent
    return agent


# Train the DQN agent
if __name__ == "__main__":

    env = TicTacToe()
    agent = DQNAgent()

    episodes = 100000000
    # controls noise in our model
    # higher batch is less noise but more space and time to run
    batch_size = 64

    agent = train_model(env, agent, episodes, batch_size)
    agent.save("dqn_tictactoe.pth")

    # Integrate with GUI
    agent.load("dqn_tictactoe.pth")
    gui = TicTacToeGUI(agent, env)
    gui.start()
