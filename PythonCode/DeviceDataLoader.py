# %%
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datsets
from torchvision.transforms import ToTensor
import os 

#Go to the parent folder AI II FINAL CODE as main path 
try: 
    import os
    os.chdir('../../AI II FINAL CODE/')
except:
    pass 

from HelperFunctions import *

#Import dataset from data folder in parent folder 
dataset = datsets.FashionMNIST(root='data/', download=False, transform=ToTensor())

#Make 40% of the data testing data 
train_indices, val_indices = split_indices(len(dataset), 0.4)

#Create the training and testing batches 
batch_size=512

#Training sampler and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, 
                        batch_size, sampler = train_sampler)

#Validation sampler and data loader
val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset, 
                        batch_size, sampler = val_sampler)

#USe cuda when avilable 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")


#Load the training and testing data 
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)


