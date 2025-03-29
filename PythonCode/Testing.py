# %%
#Import modules 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# Use a white background for matplotlib figures
matplotlib.rcParams['figure.facecolor'] = '#ffffff'

# %%
#Go to the parent folder AI II FINAL CODE as main path 
try: 
    import os
    os.chdir('../../AI II FINAL CODE/')
except:
    pass 

# %%
#import os 
str(os.getcwd())[-16:]


# %%
#Import local modules 
%run CurrentCode/Model2.ipynb
%run CurrentCode/actFunctions2.ipynb
%run CurrentCode/DeviceDataLoader2.ipynb

# %%
actFuncList = [erfMinus, erfPlus, erfPlus2, erfPlus3, erfRelu, squarePlus, nn.ReLU, nn.Softplus, nn.SELU] 
actFuncList_str = ['relu', 'selu', 'squarplus', 'softplus','erfrelu', 'erfplus'] 
print(actFunc)

# %%
#Make the model with the respective params and put it into your GPU/CPU device 
torch.manual_seed(1
                  )
model = ResNet(block=ResidualBlock, img_input_dim=28, layers=50, actFunc=actFunc)
to_device(model, device)

# %%
history = []
LR = [0.2, 0.1, 0.05, 0.01, 0.005]
#test fit the model to see if it works well 
#model.fit(epochs=1, train_loader=train_loader, val_loader=val_loader)

# %%
#Fit the model 
for i in range(len(LR)):
    history += model.fit(epochs=5, lr=LR[i], mo=0.1, train_loader=train_loader, val_loader=val_loader)

# %%


# %%
accuracies = [x['val_acc'] for x in history]
plt.plot(-np.log(1-np.array(accuracies)), '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');
print(max(-np.log(1-np.array(accuracies))))

# %%
time = [x['epoch_time'] for x in history]
plt.plot(time, '-x')
plt.xlabel('epoch')
plt.ylabel('time (s)')
plt.title('Time per epoch')


# %%
np.mean(time), np.std(time)

# %%


# %%


# %%


# %%



