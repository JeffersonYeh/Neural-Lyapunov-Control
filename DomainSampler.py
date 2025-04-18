import torch as th
import numpy as np

class DomainSampler:

    def __init__(self, N, D_in, device, lb=-6, ub=6):
        
        self.device = device
        self.x = th.Tensor(N, D_in).uniform_(lb, ub).to(dtype=th.float32, device=device)
        self.x_0 = th.zeros([1, 2], dtype=th.float32, device=device)


    def AddCounterexamples(self, CE, N, replace=False): 

        if self.x.shape[0] >= 2000 and replace:
            idx = th.randperm(self.x.shape[0], device=self.device)[:-N]
            self.x = self.x[idx]

        # Adding CE back to sample set
        c = []
        nearby= []
        for i in range(CE.size()):
            c.append(CE[i].mid())
            lb = CE[i].lb()
            ub = CE[i].ub()
            nearby_ = np.random.uniform(lb, ub, N)
            nearby.append(nearby_)

        for i in range(N):
            n_pt = []
            for j in range(self.x.shape[1]):
                n_pt.append(nearby[j][i])             
            self.x = th.cat((self.x, th.tensor([n_pt], device=self.device)), 0).to(th.float32)
        

    def get_X(self):
        return self.x
    
    def get_x0(self):
        return self.x_0
    

