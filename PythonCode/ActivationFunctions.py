# %%
import torch.nn as nn
import torch
import numpy as np
from torch import exp, where, erf, tensor
from torch import pow as POW
from numpy import pi

# %%
#https://arxiv.org/abs/2112.11687
#custom grad activation function helper for squareplus
class squarePlus_helper(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, data:tensor):

        val = POW(data, 2) + 1/4
        
        ctx.save_for_backward(data, val)

        return 0.5*(POW(val, 0.5) + data) 
        
    @staticmethod
    def backward(ctx, grad_output:tensor):

        (data, val) = ctx.saved_tensors

        grad = 0.5*(1 + data*POW(val, -0.5))
        
        return grad*grad_output


class squarePlus(nn.Module):

    def __init__(self) -> None:
        super(squarePlus, self).__init__()
        self.fn =  squarePlus_helper.apply

    def forward(self, x) -> tensor:

        return self.fn(x)

# %%


# %%
#https://arxiv.org/abs/2306.01822
#custom grad activation function helper for erf plus

class erfRelu_helper(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, data:tensor):
        
        ctx.save_for_backward(data)

        return where(data < 0, np.sqrt(pi)/2*erf(data), data)
        
    @staticmethod
    def backward(ctx, grad_output:tensor):
        
        (data, ) = ctx.saved_tensors

        grad = where(data < 0, exp(-POW(data, 2)), 1)
        
        return grad*grad_output

#Our custom grad function as a nn.Module 

class erfRelu(nn.Module):

    def __init__(self) -> None:
        super(erfRelu, self).__init__()
        self.fn = erfRelu_helper.apply

    def forward(self, x) -> tensor:

        return self.fn(x)

#Our custom grad activation function helper for erf plus
class erfPlus_helper(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, data:tensor):
        
        a = np.sqrt(np.pi)/2
        
        val = POW(data, -1)
        
        grad = where(data < 0, -erf(a*val), 1)
        
        grad_der = grad + where(data < 0, val*exp(-POW(a*val, 2)), 0)
        
        ctx.save_for_backward(grad_der)

        return data*grad
        
    @staticmethod
    def backward(ctx, grad_output:tensor):
        
        (grad, ) = ctx.saved_tensors
        
        return grad*grad_output

#Our custom grad function as a nn.Module 
class erfPlus(nn.Module):

    def __init__(self) -> None:
        super(erfPlus, self).__init__()
        self.fn = erfPlus_helper.apply

    def forward(self, x) -> tensor:

        return self.fn(x)
    
# %%
#Our custom grad activation function helper for erf minus
class erfMinus_helper(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, data:tensor):
        
        #normalization factor 
        data = 2*data +1/3 
        
        grad = where(data < 0, 0, exp(-POW(data, -2)))
        
        ctx.save_for_backward(2*grad)
    
        grad = data*grad + where(data < 0, 0, np.sqrt(pi)*erf(POW(data, -1)) - np.sqrt(pi))

        return grad
        
    @staticmethod
    def backward(ctx, grad_output:tensor):
        
        (grad, ) = ctx.saved_tensors

        return grad*grad_output

#Our custom grad function as a nn.Module 
class erfMinus(nn.Module):

    def __init__(self) -> None:
        super(erfMinus, self).__init__()
        self.fn = erfMinus_helper.apply

    def forward(self, x) -> tensor:

        return self.fn(x)
    
#Our custom grad activation function helper for erf plus
class erfPlus2_helper(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, data:tensor):
        
        a = np.pi**(-1/2)

        val = a*POW(data, -1)
        
        grad = where(data < 0, 1-exp(-POW(val, 2)), 1)
        
        ctx.save_for_backward(grad)
        
        grad = data*grad - where(data < 0, erf(val)+1, 0)

        return grad
        
    @staticmethod
    def backward(ctx, grad_output:tensor):
        
        (grad, ) = ctx.saved_tensors
        
        return grad*grad_output

#Our custom grad function as a nn.Module 
class erfPlus2(nn.Module):

    def __init__(self) -> None:
        super(erfPlus2, self).__init__()
        self.fn = erfPlus2_helper.apply

    def forward(self, x) -> tensor:

        return self.fn(x)

# %%
#Our custom grad activation function helper for erf plus
class erfPlus3_helper(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, data:tensor):
        
        grad = erf(data)*0.5 + 0.5 
        
        ctx.save_for_backward(grad)

        return data*grad + exp(-POW(data, 2))*0.5/np.sqrt(np.pi) - 0.5/np.sqrt(np.pi)
        
    @staticmethod
    def backward(ctx, grad_output:tensor):
        
        (grad, ) = ctx.saved_tensors
        
        return grad*grad_output

#Our custom grad function as a nn.Module 
class erfPlus3(nn.Module):

    def __init__(self) -> None:
        super(erfPlus3, self).__init__()
        self.fn = erfPlus3_helper.apply

    def forward(self, x) -> tensor:

        return self.fn(x)
