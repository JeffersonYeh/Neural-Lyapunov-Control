import numpy as np
import torch as th
from dreal import *
import control

class InvertedPendulum:

    def __init__(self, device):
        
        self.G = 9.81 
        self.L = 0.5  
        self.m = 0.15
        self.b1 = 0.1
        self.b2 = 0.2

        self.device = device


    def f(self, x, u):
        y = []
       
        for r in range(0,len(x)): 
            
            if x[r][0] <= 0:
                b = self.b1
            else:
                b = self.b2

            f_ = [ x[r][1], 
                (self.m * self.G * self.L * th.sin(x[r][0])- b * x[r][1]) / (self.m * self.L**2)]
            y.append(f_) 

        y = th.tensor(y, device=self.device)
        y[:,1] = y[:,1] + (u[:,0]/(self.m * self.L**2))


        return y
    

    def f1(self, x, u):
        y = []
       
        for r in range(0,len(x)): 

            f_ = [ x[r][1], 
                (self.m * self.G * self.L * th.sin(x[r][0])- self.b1 * x[r][1]) / (self.m * self.L**2)]
            y.append(f_) 

        y = th.tensor(y, device=self.device)
        y[:,1] = y[:,1] + (u[:,0]/(self.m * self.L**2))


        return y
    
    def f2(self, x, u):
        y = []
       
        for r in range(0,len(x)): 

            f_ = [ x[r][1], 
                (self.m * self.G * self.L * th.sin(x[r][0])- self.b2 * x[r][1]) / (self.m * self.L**2)]
            y.append(f_) 

        y = th.tensor(y, device=self.device)
        y[:,1] = y[:,1] + (u[:,0]/(self.m * self.L**2))


        return y


    def get_switch_condition(self, vars):
        x1, x2 = vars

        return x1 <= 0
    


    def get_fs(self, x, u):

        x1, x2 = x

        u_NN = (u.item(0)*x1 + u.item(1)*x2) 

        f1 = [ x2,
                (self.m * self. G * self.L * sin(x1) + u_NN - self.b1 * x2) /(self.m * self.L**2)]
        
        f2 = [ x2,
                (self.m * self.G * self.L * sin(x1) + u_NN - self.b2 * x2) /(self.m * self.L**2)]
        
        return f1, f2
    

    def get_lqr(self):
        lqr = th.tensor([[-23.58639732,  -5.31421063]], dtype=th.float32, device=self.device) 

        return lqr
    

    def split_samples(self, X):

        X1 = X[X[:, 0] <= 0]
        X2 = X[X[:, 0] > 0]

        return X1, X2
    

    def split_on_data(self, X, data1, data2):
        
        assert data1.shape == data2.shape
        out = th.zeros(data1.shape, device=X.device)

        for i, x in enumerate(X):

            if x[0] <= 0:
                out[i] = data1[i]
            
            else:
                out[i] = data2[i]

        return out

        


class LinearSwitchStable:

    def __init__(self, device):
        
        self.A1 = np.eye(2) * -3
        self.A2 = np.eye(2) * -2
        self.B = np.ones(2)

        self.device = device


    def f(self, x, u):

        y = th.zeros(x.shape, device=x.device)
       
        for r in range(0,len(x)): 
            
            if x[r][0] <= 0:
                A = th.tensor(self.A1, dtype=th.float32, device=x.device)
            else:
                A = th.tensor(self.A2, dtype=th.float32, device=x.device)
            
            y[r] = A @ x[r] + th.tensor(self.B, dtype=th.float32, device=x.device) * u[r]

        return y
    
    def f1(self, x, u):

        y = th.zeros(x.shape, device=x.device)
       
        for r in range(0,len(x)): 
    
            A = th.tensor(self.A1, dtype=th.float32, device=x.device)
            
            y[r] = A @ x[r] + th.tensor(self.B, dtype=th.float32, device=x.device) * u[r]

        return y
    
    def f2(self, x, u):

        y = th.zeros(x.shape, device=x.device)
       
        for r in range(0,len(x)): 
    
            A = th.tensor(self.A2, dtype=th.float32, device=x.device)
            
            y[r] = A @ x[r] + th.tensor(self.B, dtype=th.float32, device=x.device) * u[r]

        return y
    

    def get_switch_condition(self, vars):
        x1, x2 = vars

        return x1 <= 0
    

    def get_fs(self, x, u):

        x1, x2 = x

        u_NN = (u.item(0)*x1 + u.item(1)*x2) 

        f1 = self.A1 @ x - self.B * u_NN
        
        f2 = self.A2 @ x - self.B * u_NN
        
        return f1, f2
    
    
    def get_lqr(self):

        # C = np.eye(2)
        # D = 0

        # K, S, E = control.lqr(sys, Q, R)

        lqr = th.tensor([[0, 0]], dtype=th.float32, device=self.device) 

        return lqr
    

    def split_samples(self, X):

        X1 = X[X[:, 0] <= 0]
        X2 = X[X[:, 0] > 0]

        return X1, X2
    

    def split_on_data(self, X, data1, data2):
        
        assert data1.shape == data2.shape
        out = th.zeros(data1.shape, device=X.device)

        for i, x in enumerate(X):

            if x[0] <= 0:
                out[i] = data1[i]
            
            else:
                out[i] = data2[i]

        return out
    



