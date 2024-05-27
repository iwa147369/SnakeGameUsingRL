import torch
import numpy as np
from collections import deque
from snake_game_env import SnakeGame
from train import DQN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_model(filename='model.pth'):
  env = SnakeGame(render_mode='human')
  input_dim = env.observation_space.shape[0]
  output_dim = env.action_space.n
  model = DQN(input_dim, output_dim).to(device)
  model.load_state_dict(torch.load(filename))
  model.eval()
  result = []
  for episode in range(100):
    state, *_ = env.reset()
    done = False
    while not done:
      env.render()
      action = torch.argmax(model(torch.tensor(np.array(state), dtype=torch.float32).to(device))).item()
      state, _, done, *_ = env.step(action)
    print(f'Episode: {episode}, Total Reward: {env.get_score()}')
    result.append(env.get_score())
  env.reset()
  return result

if __name__ == '__main__':
  result = test_model('model400.pth')
  print('Max: ', max(result))
  print('Min: ', min(result))
  print('Average: ', sum(result) / len(result))