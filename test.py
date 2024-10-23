# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import LTSM_SNN as lsnn

from torch.utils.data import DataLoader
import santafe_dataloader as santafedata

#def loss(input, target):
 # return torch.abs(input - target).mean()

def get_stimulus(state):
  '''
  Converts a boolean element of the game state into a 2-dim vector for use as an input to the network
  vector is normalized to magnitude 1, intensity defines the maximum value of the vector elements
  '''
  if state == True:
    input = torch.tensor([[[1. ,0.]]])
  else:
    input = torch.tensor([[[0.,0.]]])

  return input

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Network Architecture
num_inputs = 2
num_hidden = 4
num_outputs = 3

# Temporal Dynamics
beta = 0.77

#initalize dataloader
santa_train = santafedata.TrailDataset("santafe_data.csv",transform=None,target_transform=None)
train_dataloader = DataLoader(santa_train, batch_size=10, shuffle=True)

inputmove, outputmove = next(iter(train_dataloader))

print(inputmove)
print(outputmove)

fsnn = lsnn.lstm_net(num_inputs, num_hidden, num_outputs,beta).to(device)

    
optimizer = torch.optim.Adam(fsnn.parameters(), lr=5e-4, betas=(0.9, 0.999))

print(fsnn.parameters)

loss_hist = []

#input = torch.tensor([[[1]]])
loss_val = 0

#loss = nn.MSELoss()
loss = nn.CrossEntropyLoss()

#targets = torch.tensor([[[89]]])

spk = get_stimulus(False)

spk_in = torch.squeeze(spk,1)
print("spk in: ",spk_in)


output, m1 = fsnn.forward(spk_in)

print("output: ",output)
for move in range(1):

  spk = get_stimulus(True)

  spk_in = torch.squeeze(spk,1)
  print("spk in: ",spk_in)


  output, m1 = fsnn.forward(spk_in)

  #print("mem rec: ",m1)
  print("spk out: ",output)

  #add = 1

#  if (torch.equal(output, torch.tensor([[[1.,0.,0.]]]))):
#     input = input + add

#   fsnn.train(True)

#   loss_val =+ loss(input, targets)
#   print("loss_val:",loss_val)

#   a = torch.tensor([[1,0,0]],dtype=torch.long)

#   #p = torch.stack([a,a])
#   #print("p size:", p.size())
#  # print("p: ", p)

#   loss_val = loss(output, a)

#   print("loss val: ", loss_val)

#   optimizer.zero_grad()
#   loss_val.backward()
#   optimizer.step()

#   #print(loss_val.grad)
#   #print(fsnn.parameters)

#   loss_hist.append(loss_val.item())

  #rand_move = torch.randint(low=0, high=3, size=(1,1))

  #print(rand_move)

plt.plot(loss_hist)
plt.show()