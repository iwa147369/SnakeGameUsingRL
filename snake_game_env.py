import numpy as np
import random
import time
from gym import Env
from gym.spaces import Discrete, Box
from snake_game.game import GameControl

UP = 90
DOWN = 270
LEFT = 180
RIGHT = 0

class SnakeGame(Env):
  def __init__(self, width=600, height=600, segment_size=15, time_interval=0.1):
    self.width = width
    self.height = height
    self.segment_size = segment_size
    self.time_interval = time_interval
    self.control = GameControl(width=self.width, height=self.height)

    self.state = self._get_state()
    self.done = False
    self.action_space = Discrete(5)
    self.observation_space = Box(low=0, high=2, shape=(self.width, self.height), dtype=np.int_)

  def _snake_postion(self):
    return np.array(self.control.snake.pos())
  
  def _food_position(self):
    return np.array(self.control.food.pos())
  
  def _direction(self):
    return np.array(self.control.snake.head.heading())

  def _get_state(self):
    snake_pos = self._snake_postion()
    food_pos = self._food_position()
    state = np.zeros((self.width, self.height))
    
    for segment in snake_pos:
      if int(segment[0]+self.width/2) >= self.width or int(segment[1]+self.height/2) >= self.height:
        continue
      state[int(segment[0]+self.width/2)][int(segment[1]+self.height/2)] = 1

    state[int(food_pos[0]+self.width/2)][int(food_pos[1]+self.height/2)] = 2
    
    return state.flatten()
  
  # Calculate reward base on the distance between snake head and food. The closer the snake is to the food, the higher the reward. 
  # If the snake eats the food, the reward is 100. If the snake hits the wall or itself, the reward is -100
  def _get_reward(self):
    snake_pos = self._snake_postion()
    food_pos = self._food_position()

    if snake_pos[0][0] > self.width/2 or snake_pos[0][0] < -self.width/2 or snake_pos[0][1] > self.height/2 or snake_pos[0][1] < -self.height/2:
      return -100
    
    for segment in snake_pos[1:]:
      if snake_pos[0][0] == segment[0] and snake_pos[0][1] == segment[1]:
        return -100
      
    snake_head = snake_pos[0]
    food = food_pos
    distance = np.sqrt((snake_head[0] - food[0])**2 + (snake_head[1] - food[1])**2)
    return 100/distance

  def reset(self):
    self.control.reset()
    self.done = False
    return self._get_state(), {}
  
  def step(self, action):
    if action == 0:
      self.control.snake.up()
    elif action == 1:
      self.control.snake.down()
    elif action == 2:
      self.control.snake.left()
    elif action == 3:
      self.control.snake.right()

    self.control.snake.move()
    self.done = self.control.handle_collision()
    
    return self._get_state(), self._get_reward(), self.done, {}

  def render(self, mode='training'):
    if mode == 'human':
      time.sleep(self.time_interval)
    self.control.screen.update()
    
  def close(self):
    self.control.exit()