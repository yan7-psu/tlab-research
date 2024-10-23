#John Yang
#Long short term memory spiking neural network with 3 layers and 1 lstm layer

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import surrogate

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import itertools

from IPython.display import HTML

num_steps = 2

# Define Network
class lstm_net(nn.Module):
    def __init__(self,num_inputs, num_hidden, num_outputs, beta):

        super().__init__()

        num_inputs = 2
        num_hidden = 4
        num_outputs = 3

        spike_grad_lstm = surrogate.straight_through_estimator()

        # initialize layers
        self.slstm1 = snn.SLSTM(num_inputs, num_hidden, spike_grad=spike_grad_lstm, threshold=0.1)

        self.fc = nn.Linear(num_hidden,num_outputs)
        self.lif = snn.Leaky(beta=beta,threshold=.4)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        syn1, mem1 = self.slstm1.init_slstm()

        mem2 = self.lif.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            spk1, syn1, mem1 = self.slstm1(x, syn1, mem1)

            #print("mem 1:", mem1)
            #print("spk 1: ",spk1)
            #print("syn1: ",syn1)
            cur2 = self.fc(spk1.flatten(1))
            spk2, mem2 = self.lif(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)
