from collections import namedtuple

directions = {'Right' : 0, 'Left' : 1, 'Up' : 2, 'Down' : 3}
Point = namedtuple('Point', 'x y')
BLOCK_SIZE = 20 
SPEED = 200

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)