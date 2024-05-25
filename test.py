import torch
import numpy as np
from collections import deque
from snake_game_env import SnakeGame
from train import DQN
from matplotlib import pyplot as plt
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_model(filename='model.pth'):
  env = SnakeGame()
  input_dim = 7
  output_dim = env.action_space.n
  model = DQN(input_dim, output_dim).to(device)
  model.load_state_dict(torch.load(filename))
  model.eval()
  result = []
  for episode in range(50):
    state, *_ = env.reset()
    done = False
    while not done:
      env.render()
      action = torch.argmax(model(torch.tensor(np.array(state), dtype=torch.float32).to(device))).item()
      state, _, done, *_ = env.step(action)
    print(f'Episode: {episode}, Total Reward: {env.get_score()}')
    result.append(env.get_score())
  return result

if __name__ == '__main__':
  result = test_model('model500.pth')
  plt.plot(result)
  plt.xlabel('Episode')
  plt.ylabel('Total Reward')
  plt.title('DQN Test')
  plt.show()
  

  


  
  