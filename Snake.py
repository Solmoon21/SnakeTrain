import random
import numpy as np
import pygame

from consts import *

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        
        self.w = w
        self.h = h

        pygame.init()
        self.font = pygame.font.Font('arial.ttf', 25)
        self.display = pygame.display.set_mode((w, h))
        #pygame.display.set_caption('Snake')

        self.clock = pygame.time.Clock()
        self.game_over = False
        self.reset()

    def reset(self):
        self.direction = 0
        self.head  = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-BLOCK_SIZE*2, self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.game_over = False
    
    def _place_food(self):
        x = (random.randint(0, self.w) // BLOCK_SIZE) * BLOCK_SIZE
        y = (random.randint(0, self.h) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x,y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, nextDirection):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        reward = self._move(nextDirection)
        done = False
        if self._is_Collision():
            done = True
            reward -= 100
            self.reset()
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, done, self.score
    
    def _is_Collision(self):
        if self.head in self.snake[1:]:
            self.game_over = True
            return True
        x , y = self.head

        if x >= self.w or x <= 0 or \
        y >= self.h or y <= 0 : \
            return True

        return False
    
    def _move(self, nextDirection):
        reward = 0
        self.direction = nextDirection

        DistanceToFood_1 = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)

        x,y = self.head.x, self.head.y

        if self.direction == 0:
            x += BLOCK_SIZE
        elif self.direction == 1:
            x -= BLOCK_SIZE
        elif self.direction == 2:
            y -= BLOCK_SIZE
        else:
            y += BLOCK_SIZE

        # if x > self.w: x = 0
        # if x < 0 : x = self.w
        # if y > self.h: y = 0
        # if y < 0 : y = self.h

        self.head = Point(x,y)
        self.snake.insert(0, self.head)

        DistanceToFood_2 = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)

        if ( DistanceToFood_1 - DistanceToFood_2 == 1): 
            reward = 1
        else:
            reward = -1

        if self.head == self.food:
            self.score += 1
            reward = 100
            self._place_food()
        else:
            self.snake.pop()

        return reward
    
    def get_random_dir(self):
        choices = ['Right', 'Left', 'Up', 'Down']

        if self.direction == directions['Right']:
            choices.remove('Left')
        
        if self.direction == directions['Left']:
            choices.remove('Right')
        
        if self.direction == directions['Up']:
            choices.remove('Down')
        
        if self.direction == directions['Down']:
            choices.remove('Up')

        return [directions[c] for c in choices]
    
    def get_conv_state(self):
        state = np.zeros((int(self.w/BLOCK_SIZE), int(self.h/BLOCK_SIZE), 3))
        for snake_cell in self.snake:
            state[int(snake_cell.x/BLOCK_SIZE) - 1, int(snake_cell.y/BLOCK_SIZE) - 1, 0] = 1
        state[int(self.head.x/BLOCK_SIZE) - 1, int(self.head.y/BLOCK_SIZE) - 1, 1] = 1
        state[int(self.food.x/BLOCK_SIZE) - 1, int(self.food.y/BLOCK_SIZE) - 1, 2] = 1
        return state
    
    def get_state(self):
        state = [int(self.food.x / BLOCK_SIZE) - int(self.head.x / BLOCK_SIZE),
               int(self.food.y / BLOCK_SIZE) - int(self.head.y / BLOCK_SIZE),
               int(any([(snake_cell.y == self.head.y - BLOCK_SIZE) for snake_cell in self.snake if
                        snake_cell.x == self.head.x])),
               int(any([(snake_cell.y == self.head.y + BLOCK_SIZE) for snake_cell in self.snake if
                        snake_cell.x == self.head.x])),
               int(any([(snake_cell.x == self.head.x - BLOCK_SIZE) for snake_cell in self.snake if
                        snake_cell.y == self.head.y])),
               int(any([(snake_cell.x == self.head.x + BLOCK_SIZE) for snake_cell in self.snake if
                        snake_cell.y == self.head.y]))]
        return np.array(state)
    
    def get_example_action(self):
        actions = []
        state = self.get_state()

        if state[0] > 0 and state[5] == 0:
            actions.append(0)
        elif state[0] < 0 and state[4] == 0:
            actions.append(1)

        if state[1] < 0 and state[2] == 0:
            actions.append(2)
        elif state[1] > 0 and state[3] == 0:
            actions.append(3)

        if not len(actions):
            actions = self.get_random_dir()
        return random.choice(actions)
    
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()