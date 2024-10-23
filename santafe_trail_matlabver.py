
import santefe_env as env

import torch
import torch.nn as nn
import snntorch as snn

from snntorch import spikegen

import numpy as np
import time
import snn as psnn

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import random

# define variables and levels for trail game
chosen_moves = 0
random_moves = 0

def get_stimulus(state):
    '''
    Converts a boolean element of the game state into a 2-dim vector for use as an input to the network
    vector is normalized to magnitude 1, intensity defines the maximum value of the vector elements
    '''
    if state == 7:
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
    
    #0 forward, 1 turn right, 2 turn left,
    command = [0, 1, 2, 8]

    if (torch.equal(spikes, torch.tensor([[[[1.,0.,0.]]]]))):
        print("FORWARD")
        return command[0]
    elif(torch.equal(spikes,torch.tensor([[[[0.,0.,1.]]]]))):
        print("R")
        return command[1]
    elif(torch.equal(spikes,torch.tensor([[[[0.,1.,0.]]]]))):
        print("L")
        return command[2] 
    else:
        print("N")
        return command[3]
    
# Network Architecture
num_inputs = 2
num_hidden = 300
num_outputs = 3

# Temporal Dynamics
beta = 0.77
threshold = 0.2

num_steps = 1
num_moves = 200
num_epochs = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float

PLOT_ON =True

# initialize the neural network
def main():

    global chosen_moves
    global random_moves

    right_moves = 0
    high_score = 0

    tottime = num_steps * num_moves
    
    fsnn = psnn.Net(num_inputs, num_hidden, num_outputs, beta, threshold).to(device)
    
    #use float for mseloss
    #loss = nn.MSELoss()

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(fsnn.parameters(), lr=5e-3, betas=(0.9, 0.999))
    loss_hist = []
    score_hist = []

    m = env.get_maze()

    if PLOT_ON:
        plt.ion()
        

    for epoch in range(num_epochs):

        
        m = env.set_sfe_trail(m)
        m, r, c, d= env.set_ant(m)
        print("THIS IS STARTING MAZE",'\n')
        print("Ant position: ", r , " ", c)
        print(m)
    

        hunger = 0

        for move in range(num_moves):

            print("THIS IS MOVE: ",move + 1)

            food = env.check_for_food(r,c,d,m)
            print("This is food value:", food)
            input = get_stimulus(food)
            stim_spk = spikegen.rate_conv(input)

            #forward pass
            # if (epoch < 10):
            #     fsnn.train(True)
            # else:
            #     fsnn.train(False)

            fsnn.train(True)

            spk_rec, mem_rec = fsnn.forward(stim_spk)    

            print("Ant does move: ", spk_rec)

            command = get_command(spk_rec)

            #feed move into santa fe environment
            r,c,d,m = env.input_res(r,c,d,command,m)

            dir = env.get_direction(d)

            print("ANT POSITION: ")
            print("ROW: ", r,"COL: ", c ,"DIRECTION: ", dir)
            
            # initialize the loss & sum over time
            # aka train the SNN
            loss_val = torch.zeros((1), dtype=dtype, device=device)

            # rand = random.randint(0,1)

            # if (rand == 1):
            #     target = torch.tensor([0.,1.,0.])
            # else:
            #     target = torch.tensor([0.,0.,1.])


            # Training via neuron output layer spikes only
            if (food == 7):
                loss_val = loss(spk_rec, torch.tensor([[[[1.,0.,0.]]]]))
            elif (hunger == 5):
                loss_val = loss(spk_rec, torch.tensor([[[[1.,0.,0.]]]]))
                hunger = 0
            else:
                loss_val = loss(spk_rec, torch.tensor([[[[0.,0.,1.]]]]))
                hunger = hunger + 1
                print("HUNGER:", hunger)

            print("LOSS VAL:", loss_val)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if PLOT_ON:
                fig1   
                
                plt.show()

            

        score = np.count_nonzero(m == 7)

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())
        score_hist.append(89-score)

    plt.plot(score_hist)
    plt.title("Food collected with Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Food Collected")
    plt.show()



if __name__ == '__main__':
    main()