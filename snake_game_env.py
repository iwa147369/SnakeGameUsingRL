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
  def __init__(self, width=600, height=600, segment_size=20, time_interval=0.1, render_mode='training'):
    super().__init__()

    self.width = width
    self.height = height
    self.segment_size = segment_size
    self.time_interval = time_interval
    self.render_mode = render_mode
    self.control = GameControl(width=self.width, height=self.height, render=self.render)
    
    self.action_space = Discrete(5)
    self.observation_space = Box(low=-max(self.width, self.height)/2, high=max(self.width, self.height), 
                                 shape=(9, ), dtype=np.float32)
    self.state = self._get_state()
    self.done = False
    
  def _snake_postion(self):
    snake_pos = []
    for segment in self.control.snake.pos():
      snake_pos.append((segment[0], segment[1]))
    return np.array(snake_pos)
  
  def _food_position(self):
    food_pos = self.control.food.pos()
    return np.array(food_pos)
  
  def _direction(self):
    return self.control.snake.head.heading()

  def _get_state(self):
    snake_pos = self._snake_postion()
    food_pos = self._food_position()

    def is_left_collision():
      head = snake_pos[0]
      if head[0] - 1.2 * self.segment_size <= -self.width:
        return True 
      for segment in snake_pos[1:]:
        if head[0] - 1.2 * self.segment_size <= segment[0] and head[1] == segment[1]:
          return True
      return False  
      
    def is_right_collision():
      head = snake_pos[0]
      if head[0] + 1.2 * self.segment_size >= self.width:
        return True 
      for segment in snake_pos[1:]:
        if head[0] + 1.2 * self.segment_size >= segment[0] and head[1] == segment[1]:
          return True
      return False
      
    def is_up_collision():
      head = snake_pos[0]
      if head[1] + 1.2 * self.segment_size >= self.height:
        return True 
      for segment in snake_pos[1:]:
        if head[0] == segment[0] and head[1] + 1.2 * self.segment_size >= segment[1]:
          return True
      return False
      
    def is_down_collision():
      head = snake_pos[0]
      if head[1] - 1.2 * self.segment_size <= -self.height:
        return True 
      for segment in snake_pos[1:]:
        if head[0] == segment[0] and head[1] - 1.2 * self.segment_size <= segment[1]:
          return True
      return False
    
    def is_food_left():
      head = snake_pos[0]
      return food_pos[0] < head[0]
  
    def is_food_right():
      head = snake_pos[0]
      return food_pos[0] > head[0]
    
    def is_food_up():
      head = snake_pos[0]
      return food_pos[1] > head[1]
    
    def is_food_down():
      head = snake_pos[0]
      return food_pos[1] < head[1]
    
    state = [
      is_food_left(),
      is_food_right(),
      is_food_up(),
      is_food_down(),
      is_left_collision(),
      is_right_collision(),
      is_up_collision(),
      is_down_collision(),
      self._direction()/90
    ]

    return state

  def reset(self):
    self.control.reset()
    self.done = False
    self.frame_iteration = 0
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
    status, self.done = self.control.handle_collision()
    
    # Reward 10 if the snake eats the food, -10 if the snake collisdes with itself, -5 if the snake collides with the wall, 
    # and -1/(length of snake) if the snake moves each 20 frames to prevent the snake from moving in circles
    if status == 2:
      return self._get_state(), 10, self.done, {}
    elif status == 0:
      return self._get_state(), -10, self.done, {}
    elif status == 1:
      return self._get_state(), -5, self.done, {}
    else:
      return self._get_state(), 0, self.done, {}
    
  def render(self):
    if self.render_mode == 'human':
      time.sleep(self.time_interval)
    self.control.screen.update()
    
  def get_score(self):
    return self.control.scoreboard.score

  def close(self):
    self.control.exit()