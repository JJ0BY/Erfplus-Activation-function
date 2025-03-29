# %%
#Import modules 
import time as T

#Go to the parent folder AI II FINAL CODE as main path 
try: 
    import os
    if str(os.getcwd())[-16:] != "AI II FINAL CODE": 
        os.chdir('../../AI II FINAL CODE/')
except:
    pass 

from HelperFunctions import *
from Model import *
from ActivationFunctions import *

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datsets
from torchvision.transforms import ToTensor
import os 

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


# %%
trialsPerExperiment = 5
#choose random seeds for trials 
seeds = [12334, 12341, 13424, 43124, 1413431, 1341, 132, 1765, 98786, 5634]

LR = [0.2, 0.1, 0.05, 0.025, 0.01]

actFuncList = [nn.ReLU, nn.GELU, squarePlus,  nn.Softplus, erfRelu, erfPlus, erfPlus2] 
actFuncList_str = ['relu', 'gelu', 'squarplus', 'softplus','erfrelu', 'erfplus', 'erfPlus2'] 
histories = [] 

#Check if it is correct model 

print('Start Experiment') 

totTime0 = T.time()

for ii in range(trialsPerExperiment): 
    SEED = np.random.randint(1,99999)
    batches = []
    for actFunc_num in range(len(actFuncList)):
        torch.manual_seed(SEED)
        history = []
        t0 = T.time()
        #Make the model with the respective params and put it into your GPU/CPU device 
        model = ResNet(block=ResidualBlock, img_input_dim=28, layers=50, actFunc=actFuncList[actFunc_num])
        model = to_device(model, device)
        #Fit the model 
        for jj in range(len(LR)):
            history += fit(model, epochs=5, lr=LR[jj], mo=0.1, train_loader=train_loader, val_loader=val_loader, print_statement=False)
        
        score = -np.log(1-history[-1]['val_acc'])
        
        print(f'Trial: {ii+1} Time (s): {T.time()-t0:.3f} Score: {score:.3f} AF: {str(actFuncList_str[actFunc_num]) }')
        
        batches.append(history)
        
    histories.append(batches)
    
    
totTime1 = T.time()

print(f"Experiment run time: {((totTime1-totTime0)/60):.3f} mins") 

NEWdata = True 

if NEWdata: 
    uploadObject(histories, 'Results/Experiment1_pkl', nosilence=True)
else:
    try: 
        hist_data = loadObject('Results/Experiment1_pkl')
        
        uploadObject(histories + hist_data, 'Results/Experiment1_pkl', nosilence=True)
    except:
        uploadObject(histories, 'Results/Experiment1_pkl', nosilence=True)