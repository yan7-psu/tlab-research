#John Yang
#Feed forward spiking neural network with 3 layers

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import itertools

from IPython.display import HTML

num_steps = 6

# Define Network
class Net(nn.Module):
    def __init__(self,num_inputs, num_hidden, num_outputs, beta, threshold):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            print("mem1: ",mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec,dim=0), torch.stack(mem2_rec,dim=0) # time-steps x batch x num_out
    

    

ffssn = Net(2,4,3,.95,0.4)

input = torch.tensor([[1.,1.]])


#we see that membrane potential rises and resets when num_steps is longer than 1
output, mem2 = ffssn(input)