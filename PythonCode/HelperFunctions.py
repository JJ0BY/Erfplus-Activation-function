# %%
import time as T 
import numpy as np
import pickle as P
import os 
import torch 

#Go to the parent folder AI II FINAL CODE as main path 
try: 
    import os
    os.chdir('../../AI II FINAL CODE/')
except:
    pass 

# %%
#Pickle based upload and load python file helper function  

def uploadObject(Object, filename, nosilence=True):
    t0 = T.time()
    i=1
    while 1==1:
        try:
            filehandler = open(filename, 'wb') 
            
            P.dump(Object, filehandler)
            loadObject(filename)
            if nosilence:
                print('----------------FINISHED SAVING----------------')
            break
        except EOFError:
            if nosilence:
                print(f'Attempt #{i} failed')
            i+=1
            if i == 11:
                if nosilence:
                    print('Failed to Save')
                break
            
        
    t1 = T.time() - t0
    if nosilence:
        print(f'Time: {t1-t0:.5f} s\n')
    
def loadObject(filename, nosilence=True):
    t0 = T.time()
    infile = open(filename,'rb')
    Object = P.load(infile)
    infile.close()
    t1 = T.time()
    if nosilence:
        print(f'Time: {t1-t0:.5f} s\n')
    return Object

# %%
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# %%
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(model, dl, device):
        model.dl = dl
        model.device = device
        
    def __iter__(model):
        """Yield a batch of data after moving it to device"""
        for b in model.dl: 
            yield to_device(b, model.device)

    def __len__(model):
        """Number of batches"""
        return len(model.dl)

# %%
#Split the train and test indicies 
def split_indices(n,val_pct):
    #determine size of validation set
    n_val = int(n*val_pct)
    #create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    #Pick first n_val random indices as validation set
    return idxs[n_val:], idxs[:n_val]
# %%


def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(model, outputs)

def fit(model, epochs, lr, mo, train_loader, val_loader, opt_func=torch.optim.SGD, print_statement=True):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr, mo)
    for epoch in range(epochs):
        t0 = T.time() 
        # Training Phase 
        for batch in train_loader:
            loss = training_step(model, batch)
            loss.backward()
            #loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['epoch_time'] = T.time() - t0 
        if print_statement:
            epoch_end(model, epoch, result)
        history.append(result)
    if print_statement:
        print('-----------------------------------------------------')
    return history

def training_step(model, batch):
    images, labels = batch 
    out = model.forward(images)                  # Generate predictions
    loss = model.criterion(out, labels) # Calculate loss
    return loss

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validation_step(model, batch):
    images, labels = batch 
    out = model.forward(images)                 # Generate predictions
    loss = model.criterion(out, labels)         # Calculate loss
    acc = accuracy(out, labels)           # Calculate accuracy
    return {'val_loss': loss, 'val_acc': acc}
    
def validation_epoch_end(model, outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def epoch_end(model, epoch, result):
    print("Epoch [{}] val_loss: {:.4f}, val_acc: {:.4f}, time: {:.4f} s".format(epoch, result['val_loss'], result['val_acc'], result['epoch_time']))

