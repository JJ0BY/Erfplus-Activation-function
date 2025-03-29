# %%
import torch
import torch.nn as nn
import time as t 



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, actFunc=nn.ReLU, stride = 1):
        super(ResidualBlock, self).__init__()
        
        self.FC1 = nn.Sequential(
                        nn.BatchNorm1d(out_channels),
                        nn.Linear(in_channels, out_channels),
                        actFunc())
        self.FC2 = nn.Sequential(
                        nn.BatchNorm1d(out_channels),
                        nn.Linear(out_channels, out_channels)
                        )
        self.actFunc1 = actFunc()
        self.out_channels = out_channels
        #self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        out = self.FC1(x)
        out = self.FC2(out)
        out += residual
        #out = self.dropout(out)
        out = self.actFunc1(out)
        return out

# %%


# %%
class ResNet(nn.Module):
    def __init__(self, block, layers, img_input_dim = 64, actFunc=nn.ReLU, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        #main activation function 
        self.actFunc = actFunc
        
        #layers 
        self.encoder = nn.Linear(img_input_dim*img_input_dim, self.inplanes)
        self.hid_layers = self._make_layer(block, self.inplanes, layers, stride = 1)
        self.decoder = nn.Linear(self.inplanes, num_classes)
        self.output_actFunc = nn.Softmax(dim=1)
        self.input_actFunc = actFunc()
        #Loss function 
        self.criterion = nn.CrossEntropyLoss()
        
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, self.actFunc, stride))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.actFunc))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.input_actFunc(x)
        x = self.hid_layers(x)
        x = self.decoder(x)
        x = self.output_actFunc(x)
        
        return x