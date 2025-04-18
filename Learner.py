import torch as th
import torch.nn.functional as F
import numpy as np
from dreal import *


def dtanh(s):
    # Derivative of activation
    return 1.0 - s**2



class Learner(th.nn.Module):
    
    def __init__(self, n_input, n_hidden, n_output, controller=False, lqr=None):
        super(Learner, self).__init__()
        # th.manual_seed(2)
        self.layer1 = th.nn.Linear(n_input, n_hidden)
        self.activation1 = th.nn.Tanh()
        self.layer2 = th.nn.Linear(n_hidden,n_output)
        self.activation2 = th.nn.Tanh()

        self.controller = controller
        if self.controller:

            if lqr is None: raise ValueError("Input controller initialization")

            self.control = th.nn.Linear(n_input,1,bias=False)
            self.control.weight = th.nn.Parameter(lqr)


    def forward(self,x):

        h_1 = self.activation1(self.layer1(x))
        out = self.activation2(self.layer2(h_1))

        if self.controller:
            u = self.control(x)
            return out,u
        
        else:
            return out


    def get_nn_weights(self):
        w1 = self.layer1.weight.data.cpu().numpy()
        w2 = self.layer2.weight.data.cpu().numpy()
        b1 = self.layer1.bias.data.cpu().numpy()
        b2 = self.layer2.bias.data.cpu().numpy()
        

        return w1, w2, b1, b2
    
    def get_controller_weights(self):
        q = self.control.weight.data.cpu().numpy()
        
        return q


    def get_candidate(self, vars):

        w1, w2, b1, b2 = self.get_nn_weights()
    
        z1 = np.dot(vars, w1.T) + b1

        a1 = []
        
        for j in range(0,len(z1)):
            a1.append(tanh(z1[j]))

        z2 = np.dot(a1,w2.T)+b2

        V_learn = tanh(z2.item(0))

        return V_learn
    

    def get_lie_derivative(self, V_candidate, X, f):

        L_v = th.diagonal(th.mm(th.mm(th.mm(dtanh(V_candidate), self.layer2.weight)\
                          *dtanh(th.tanh(th.mm(X ,self.layer1.weight.t())+self.layer1.bias)),self.layer1.weight), f.t()), 0)
        
        return L_v
    

    def export_model(self):
        th.save(self.state_dict(), 'export/model_weights.pth')

    def import_model(self):
        self.load_state_dict(th.load('export/model_weights.pth'))
        
    

    




