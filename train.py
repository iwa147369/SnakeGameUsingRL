import gym
import random
import torch
import numpy as np
from collections import deque
from snake_game_env import SnakeGame

learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 20000
batch_size    = 4096

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
  def __init__(self, buffer_size):
    self.buffer_size = buffer_size
    self.buffer = deque(maxlen=buffer_size)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size):
    if batch_size > len(self.buffer):
      return None
    else:
      return random.sample(self.buffer, batch_size)

  def size(self):
    return len(self.buffer)
  
class DQN(torch.nn.Module):
  def __init__(self, input_dim, output_dim):
    super(DQN, self).__init__()
    self.fc1 = torch.nn.Linear(input_dim, 128)
    self.fc2 = torch.nn.Linear(128, 128)
    self.fc3 = torch.nn.Linear(128, output_dim)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
  def sample_action(self, state, epsilon, max_action=2):
    if random.random() < epsilon:
      return random.randint(0, max_action - 1)
    else:
      state = torch.tensor(np.array(state), dtype=torch.float32).to(device)
      return torch.argmax(self.forward(state)).item()

def train_model(model, target_model, buffer, optimizer, gamma, batch_size):
  if buffer.size() < batch_size:
    return
  batch = buffer.sample(batch_size)
  states, actions, rewards, next_states, dones = zip(*batch)
  states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
  actions = torch.tensor(np.array(actions), dtype=torch.int).to(device)
  rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
  next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
  dones = torch.tensor(np.array(dones), dtype=torch.bool).to(device)

  q_values = model(states)
  next_q_values = target_model(next_states)
  next_q_values[dones == 1] = 0
  target_q_values = rewards + gamma * torch.max(next_q_values, dim=1).values

  loss = torch.nn.functional.mse_loss(q_values[range(batch_size), actions], target_q_values)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

def main():
  env = SnakeGame()
  input_dim = env.observation_space.shape[0]
  output_dim = env.action_space.n
  model = DQN(input_dim, output_dim).to(device)
  target_model = DQN(input_dim, output_dim).to(device)
  target_model.load_state_dict(model.state_dict())
  buffer = ReplayBuffer(buffer_limit)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  epsilon = 1
  epsilon_decay = 0.997
  min_epsilon = 0.1
  update_target_rate = 10

  min_reward = 0
  for episode in range(1001):
    state, *_ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
      action = model.sample_action(state, epsilon, output_dim)
      env.render()
      next_state, reward, done, *_ = env.step(action)
      buffer.add(state, action, reward, next_state, done)
      train_model(model, target_model, buffer, optimizer, gamma, batch_size)
      state = next_state
      total_reward += reward
    epsilon = max(epsilon * epsilon_decay, min_epsilon)
    if episode % update_target_rate == 0 and episode > 0:
      target_model.load_state_dict(model.state_dict())
      print(f'Episode: {episode}, Average reward: {total_reward / update_target_rate:.2f}, epsilon: {epsilon:.2}')
      print('Target model updated')
    
    if total_reward > min_reward:
      min_reward = total_reward
      torch.save(model.state_dict(), f'model{str(episode)}_{str(total_reward)}.pth')

    if episode % 100 == 0 and episode > 0:
      torch.save(model.state_dict(), f'model{str(episode)}_{str(total_reward)}.pth')

if __name__ == '__main__':
  main()