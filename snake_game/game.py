from turtle import Screen
from .snake import Snake
from .food import Food
from .scoreboard import Scoreboard
import time

class GameControl():
    def __init__(self, width=600, height=600, time_interval=0.1, segment_size=15, render=False):
        self.width = width
        self.height = height
        self.time_interval = time_interval
        self.segment_size = segment_size

        self.snake = Snake()
        self.food = Food()
        self.scoreboard = Scoreboard()
        self.screen = Screen()
        self.screen.setup(width=self.width, height=self.height)
        self.screen.bgcolor("black")
        self.screen.title("My Snake Game")
        self.screen.tracer(0)
        self.screen.listen()
        self.screen.onkey(self.snake.up, "Up")
        self.screen.onkey(self.snake.down, "Down")
        self.screen.onkey(self.snake.left, "Left")
        self.screen.onkey(self.snake.right, "Right")

        self.game_is_on = True

    def run(self):
        while self.game_is_on:
            self.screen.update()
            time.sleep(self.time_interval)
            self.snake.move()

            self.handle_collision() 

    # Handle collision with food, wall and tail
    # Return status and game over status
    def handle_collision(self):
        status = -1
        # Detect collision with food.
        if self.snake.head.distance(self.food) < self.segment_size:
            self.food.refresh()
            self.snake.extend()
            self.scoreboard.increase_score()
            status = 2

        # Detect collision with wall.
        if self.snake.head.xcor() > (self.width/2 - self.segment_size) or self.snake.head.xcor() < (-self.width/2 + self.segment_size) or self.snake.head.ycor() > (self.height/2 - self.segment_size) or self.snake.head.ycor() <  (-self.height/2 + self.segment_size):
            self.game_is_on = False
            self.scoreboard.game_over()
            return (1, True)

        # Detect collision with tail.
        for segment in self.snake.segments:
            if segment == self.snake.head:
                pass
            elif self.snake.head.distance(segment) < self.segment_size:
                self.game_is_on = False
                self.scoreboard.game_over()
                return (0, True)
            
        return (status, False)

    # FIXME: Current snake.reset() function does not destroy the snake object, 
    # so the snake object is still in the memory
    def reset(self):
        self.snake.reset()
        self.food.refresh()
        self.scoreboard.reset()
        self.game_is_on = True
        
    def exit(self):
        self.reset()
        self.screen.exitonclick()
        self.screen.bye()
