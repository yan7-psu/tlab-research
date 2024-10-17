import pygame
from simulator.TrailGame.game import Game, Controller
from simulator.TrailGame.environment import Map, Trail
from simulator.TrailGame.ant import Ant

import torch
import torch.nn as nn
import snntorch as snn

from snntorch import spikegen

import numpy as np
import time
import snn as psnn

import matplotlib.pyplot as plt

import random

# define variables and levels for trail game

SANTA_FE = [[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [60.0, 0.0], [60.0, 20.0], [60.0, 40.0], [60.0, 60.0], [60.0, 80.0],
        [60.0, 100.0], [80.0, 100.0], [100.0, 100.0], [120.0, 100.0], [160.0, 100.0], [180.0, 100.0], [200.0, 100.0],
        [220.0, 100.0], [240.0, 100.0], [240.0, 120.0], [240.0, 140.0], [240.0, 160.0], [240.0, 180.0], [240.0, 200.0],
        [240.0, 240.0], [240.0, 260.0], [240.0, 280.0], [240.0, 300.0], [240.0, 360.0], [240.0, 380.0], [240.0, 400.0],
        [240.0, 420.0], [240.0, 440.0], [240.0, 460.0], [220.0, 480.0], [200.0, 480.0], [180.0, 480.0], [160.0, 480.0],
        [140.0, 480.0], [100.0, 480.0], [80.0, 480.0], [40.0, 500.0], [40.0, 520.0], [40.0, 540.0], [40.0, 560.0],
        [60.0, 600.0], [80.0, 600.0], [100.0, 600.0], [120.0, 600.0], [160.0, 580.0], [160.0, 560.0], [180.0, 540.0],
        [200.0, 540.0], [220.0, 540.0], [240.0, 540.0], [260.0, 540.0], [280.0, 540.0], [300.0, 540.0], [340.0, 520.0],
        [340.0, 500.0], [340.0, 480.0], [340.0, 420.0], [340.0, 380.0], [340.0, 360.0], [340.0, 340.0], [360.0, 320.0],
        [420.0, 300.0], [420.0, 280.0], [420.0, 220.0], [420.0, 200.0], [420.0, 180.0], [420.0, 160.0], [440.0, 100.0],
        [460.0, 100.0], [500.0, 80.0], [500.0, 60.0], [520.0, 40.0], [540.0, 40.0], [560.0, 40.0], [580.0, 60.0],
        [580.0, 80.0], [580.0, 120.0], [580.0, 180.0], [580.0, 240.0], [560.0, 280.0], [540.0, 280.0], [520.0, 280.0],
        [460.0, 300.0], [480.0, 360.0], [540.0, 380.0], [520.0, 440.0], [460.0, 460.0]]

WIDTH = 640
HEIGHT = 640

chosen_moves = 0
random_moves = 0

def get_stimulus(state,intensity = 1):
    '''
    Converts a boolean element of the game state into a 2-dim vector for use as an input to the network
    vector is normalized to magnitude 1, intensity defines the maximum value of the vector elements
    '''
    if state == True:
        input = torch.tensor([[[1. ,0.]]])
    else:
        input = torch.tensor([[[0.,1.]]])

    return input

def get_command(spikes):
    '''
    Uses the spiked neuron associated with a move 
    '''
    global chosen_moves
    global random_moves
    
    command = ['f', 'l', 'r', 'n']

    if (torch.equal(spikes, torch.tensor([[[[1.,0.,0.]]]]))):
        print("FORWARD")
        return command[0]
    elif(torch.equal(spikes,torch.tensor([[[[0.,1.,0.]]]]))):
        print("L")
        return command[1]
    elif(torch.equal(spikes,torch.tensor([[[[0.,0.,1.]]]]))):
        return command[2]
        print("R")
    else:
        return command[3]
        print("N")
    
def dist_to_food(ant_pos, pellet_pos):

    delta_x = abs(ant_pos[0] - pellet_pos[0])

    if ant_pos[0] < pellet_pos[0]:
        toroid_x = ant_pos[0] + (WIDTH - pellet_pos[0])

    elif ant_pos[0] > pellet_pos[0]:
        toroid_x = pellet_pos[0] + (WIDTH - ant_pos[0])
    else:   
        toroid_x = 0

    delta_y = abs(ant_pos[1] - pellet_pos[1])

    if ant_pos[1] < pellet_pos[1]:

        toroid_y = ant_pos[1] + (HEIGHT - pellet_pos[1])

    elif ant_pos[1] > pellet_pos[1]:

        toroid_y = pellet_pos[1] + (HEIGHT - ant_pos[1])
    else:

        toroid_y = 0

    dist = min(delta_x, toroid_x) + min(delta_y, toroid_y)
    
    return dist

# Network Architecture
num_inputs = 2
num_hidden = 300
num_outputs = 3

# Temporal Dynamics
beta = 0.95
threshold = 0.2

num_steps = 1
num_moves = 1000
num_epochs = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float

# initialize the neural network
def main():

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    ant = Ant()
    game = Game()
    map_ = Map()

    
    global chosen_moves
    global random_moves

    right_moves = 0
    high_score = 0

    tottime = num_steps * num_moves
    
    fsnn = psnn.Net(num_inputs, num_hidden, num_outputs, beta, threshold).to(device)
    
    loss = nn.MSELoss()
    #loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(fsnn.parameters(), lr=1e-3, betas=(0.9, 0.999))
    loss_hist = []

    for epoch in range(num_epochs):
        
        trail = Trail()
        
        trail.load(SANTA_FE)
        game.update(ant, trail, map_)
        game.draw_screen(screen, ant, trail, map_)
        pygame.display.update()

        old_crit = 0
        
        chosen_moves = 0
        random_moves = 0
        hunger = 0
        consec_food = 0
        noops = 0
        moves_s = 0

        movetime = []   # benchmarking runtime performance
                
        if right_moves > high_score:
            high_score = right_moves
            #go = input("Testing simulation. Execute?")
            #if go == 'y':
                #testing = True
                
        next_pellet = trail.pellets[0].position # update next pellet
        dist = dist_to_food(ant.position, next_pellet)

        for move in range(num_moves):

            pygame.event.pump()
            start = time.time()

            if (move == 0):
                command = 'f'
                game.play(ant, command, command=True)
            

            food = ant.sees_food_ahead
            print("Ant see food?" ,food, "at move: ", move)
            stimulus = get_stimulus(food)
 
            #print("stimulus: ", stimulus)
            stim_spk = spikegen.rate_conv(stimulus)

            #print("input spike: ", stim_spk)

            #forward pass
            fsnn.train(True)
            spk_rec, mem_rec = fsnn.forward(stim_spk)
            print("output: ",spk_rec)

            command = get_command(spk_rec)

            #update game state
            game.play(ant, command, command=True)
            game.update(ant, trail, map_)
            game.draw_screen(screen, ant, trail, map_)
            pygame.display.update()

            # initialize the loss & sum over time
            # aka train the SNN
            loss_val = torch.zeros((1), dtype=dtype, device=device)

            score = game.food_eaten

            #print(score)

            #input = torch.tensor([[[float(score)]]],requires_grad=True)
            #target = torch.tensor([[[89.]]])
            #loss_val += loss(input ,target)
            #print("loss val: ",loss_val)

            rand = random.randint(0,1)

            if (rand == 1):
                target = torch.tensor([0.,1.,0.])
            else:
                target = torch.tensor([0.,0.,1.])


            if (food == True):
                loss_val = loss(spk_rec, torch.tensor([1.,0.,0.]))
            else:
                if (hunger == 5):
                    loss_val = loss(spk_rec, torch.tensor([1.,0.,0.]))
                    hunger = 0
                
                loss_val = loss(spk_rec, torch.tensor([0.,0.,1.]))
                hunger = hunger + 1


            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # -----------------------------------------------
            
            # if ant.was_fed:
                
            #     new_dist = dist_to_food(ant.position, next_pellet)
            #     inc = new_dist - dist
                
            #     if inc > 0:
            #         criticism = -1

            #     else:

            #         criticism = 1

            #         if game.food_eaten == 8:
            #             criticism = 100
            #         else:
            #             next_pellet = trail.pellets[0].position # update next pellet
            #             dist = dist_to_food(ant.position, next_pellet)

                
            # else:

            #     new_dist = dist_to_food(ant.position, next_pellet)
            #     inc = new_dist - dist


            #     dist = new_dist
                
            #     if inc > 0:

            #         criticism = -1

            #     elif inc < 0:

            #         criticism = 1

            #     else: # inc = 0
                    
            #         criticism = 0
            # # -----------------------------------------------
            # end = time.time() - start
            # #movetime.append(end)
            # # ------------------------------------------------

            # if criticism < 0:
            #     right_moves = move + 1
            #     break
        
        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        #print stats at end of epoch
        print(f"Epoch: {epoch}")
        print(f"Moves made: {right_moves}. No-ops: {noops}. Food collected: {game.food_eaten}\n")

        game.reset(ant, map_, trail)
        game.draw_screen(screen, ant, trail, map_)
        pygame.display.update()

    # save parameters after training
    plt.plot(loss_hist)
    #print(loss_hist.grad())

if __name__ == '__main__':
    main()