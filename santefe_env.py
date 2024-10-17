#John Yang
#Santa Fe Trail environment

import numpy as np
import matplotlib.pyplot as plt

######  constants ############################################################
height = 32
width = 32

n_input = height * width      # input dimensionality i.e. game board size

up = 0                        # which direction are we facing
right = 1
down = 2
left = 3

pellet = 7.
blank = -1.

np.set_printoptions(threshold=np.inf)   # lets us print it all to screen

def get_maze():
    grid = np.zeros((width, height))

    for i in range(0, width):
        for j in range(0, height):
            grid[i][j] = blank
        
    return grid

#Easy test: set food pellets in a horizontal straight line
def set_food_inline(grid,width):
    line = np.full(width, pellet)
    grid[2] = line
    return grid

def set_sfe_trail(grid):
    grid[0,0] = pellet
    grid[0,1] = pellet
    grid[0,2] = pellet
    grid[0,3] = pellet

    grid[1,3] = pellet
    grid[2,3] = pellet
    grid[3,3] = pellet
    grid[4,3] = pellet
    grid[5,3] = pellet

    grid[5,4] = pellet
    grid[5,5] = pellet
    grid[5,6] = pellet

    grid[5,8] = pellet
    grid[5,9] = pellet
    grid[5,10] = pellet
    grid[5,11] = pellet
    grid[5,12] = pellet

    grid[6,12] = pellet
    grid[7,12] = pellet
    grid[8,12] = pellet
    grid[9,12] = pellet
    grid[10,12] = pellet

    grid[12,12] = pellet
    grid[13,12] = pellet
    grid[14,12] = pellet
    grid[15,12] = pellet

    grid[18,12] = pellet
    grid[19,12] = pellet
    grid[20,12] = pellet
    grid[21,12] = pellet
    grid[22,12] = pellet
    grid[23,12] = pellet

    grid[24,11] = pellet
    grid[24,10] = pellet
    grid[24,9] = pellet
    grid[24,8] = pellet
    grid[24,7] = pellet

    grid[24,4] = pellet
    grid[24,3] = pellet

    grid[25,1] = pellet
    grid[26,1] = pellet
    grid[27,1] = pellet
    grid[28,1] = pellet

    grid[30,2] = pellet
    grid[30,3] = pellet
    grid[30,4] = pellet
    grid[30,5] = pellet

    grid[29,7] = pellet
    grid[28,7] = pellet

    grid[27,8] = pellet
    grid[27,9] = pellet
    grid[27,10] = pellet
    grid[27,11] = pellet
    grid[27,12] = pellet
    grid[27,13] = pellet
    grid[27,14] = pellet

    grid[26,16] = pellet
    grid[25,16] = pellet
    grid[24,16] = pellet

    grid[21,16] = pellet

    grid[19,16] = pellet
    grid[18,16] = pellet
    grid[17,16] = pellet

    grid[16,17] = pellet

    grid[15,20] = pellet
    grid[14,20] = pellet

    grid[11,20] = pellet
    grid[10,20] = pellet
    grid[9,20] = pellet
    grid[8,20] = pellet

    grid[5,21] = pellet
    grid[5,22] = pellet

    grid[4,24] = pellet
    grid[3,24] = pellet

    grid[2,25] = pellet
    grid[2,26] = pellet
    grid[2,27] = pellet

    grid[3,29] = pellet
    grid[4,29] = pellet

    grid[6,29] = pellet

    grid[9,29] = pellet

    grid[12,29] = pellet

    grid[14,28] = pellet
    grid[14,27] = pellet
    grid[14,26] = pellet

    grid[15,23] = pellet

    grid[18,24] = pellet

    grid[19,27] = pellet

    grid[22,26] = pellet

    grid[23,23] = pellet
    return grid

#place ant into startin position
def set_ant(grid):
  d = 1

  r = np.argwhere(grid == pellet)
  grid[r[0,0],r[0,1]] = d
  
  return grid,r[0,0],r[0,1], d

def hi_score(grid):
  score = np.argwhere(grid == pellet)
  num_row, num_col = score.shape
  
  return num_row

# SantaFe rules say look at next block in the direction we are currently facing
# peak to see if food is there?
def check_for_food(r, c, d, grid):

  food = 0        # don't know if food, take a peak

  if d == up:
    if r > 0:  
      food = grid[r-1,c]
      return food

  if d == right:
    if c < (width-1):  
      food = grid[r,c+1]
      return food

  if d == down:
    if r < (height-1):  
      food = grid[r+1,c]
      return food

  if d == left:
    if c > 0:   
      food = grid[r,c-1]
      return food
  
  return np.array([-1.0]) 

def move_ant(r, c, d, grid):
    if d == 0:                              # move up
        if r > 0:
            grid[r,c] = blank
            r = r-1
            grid[r,c] = d
            return grid,r,c
        else: 
            grid[r,c] = d
            return grid,r,c
    elif d == 1:                            # move right
        if c < (width-1):           # check if move still inside array edges
            grid[r,c] = blank
            c = c+1
            grid[r,c] = d
            return grid,r,c
        else:                   # else stay put
            grid[r,c] = d
            return grid,r,c
    elif d == 2:                            # move down
        if r < (height-1): 
            grid[r,c] = blank
            r = r+1
            grid[r,c] = d
            return grid,r,c
        else: 
            grid[r,c] = d
            return grid,r,c
    elif d == 3:                            # move left
        if c > 0: 
            grid[r,c] = blank
            c = c-1
            grid[r,c] = d
            return grid,r,c        
        else: 
            grid[r,c] = d
            return grid,r,c


def get_direction(d):
    if d == 0: return "up"
    if d == 1: return "right"
    if d == 2: return "down"
    if d == 3: return "left"

#output data after recieving data from reservoir
#this function will change ant direction or move it
def input_res(r,c,d,move,maze):
  #go forward
  if move == 0:
    m,r,c = move_ant(r,c,d,maze)
    return r,c,d,maze
  
  #turn right
  if move == 1 and d < 3:
    ant = maze[r,c]
    print("ant rc: ",ant)
    ant = ant + 1
    print("ant",ant)
    maze[r,c] = ant
    d = ant
  elif move == 1 and d >= 3:
    maze[r,c] = 0
    d = maze[r,c]
  #turn left
  elif move == 2 and d > 0:
    ant = maze[r,c]
    print("ant rc: ",ant)
    ant = ant - 1
    print("ant",ant)
    maze[r,c] = ant
    d = ant
  elif move == 2 and d ==0:
    maze[r,c] = 3
    d = maze[r,c]
  else:
    print()
  
  d = int(d)
  return r,c,d,maze

def check_finish(maze):
  check = np.argwhere(maze == pellet)
  if check.size == 0:
    return 1
  
  return 0

m = get_maze()
m = set_sfe_trail(m)
m, r, c, d= set_ant(m)

print("THIS IS STARTING MAZE",'\n')
print("Ant position: ", r , " ", c)
plt.matshow(m)
plt.show()

print(np.count_nonzero(m == pellet))
