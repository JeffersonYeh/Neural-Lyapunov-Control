
import torch as th
import torch.nn.functional as F
import numpy as np
from dreal import *
import timeit 
import matplotlib.pyplot as plt


from Learner import Learner
from Falsifier import Falsifier
from Dynamics import InvertedPendulum, LinearSwitchStable
from DomainSampler import DomainSampler
from Plotter import Plotter, export_learning_curve


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}") 


## Parameters

N = 500             # sample size
D_in = 2            # input dimension
H1 = 6              # hidden dimension
D_out = 1           # output dimension

LR = 0.01

out_iters = 0
TRIES = 1
MAX_ITERS = 2000


##

system = InvertedPendulum(device)
# system = LinearSwitchStable(device)

data = DomainSampler(N, D_in, device)
falsifier = Falsifier()
plotter = Plotter(device, fidelity=100)

valid = False


lyap_risks = np.zeros((TRIES, MAX_ITERS))

while out_iters < TRIES and not valid: 
    start = timeit.default_timer()

    model = Learner(D_in, H1, D_out, controller=True, lqr=system.get_lqr()).to(device)
    optimizer = th.optim.RMSprop(model.parameters(), lr=LR, alpha=0.99, eps=1e-8, weight_decay=1e-4)

    plotter.reset_plots()

    L = []
    i = 0 
    t = 0
    
    # optimizer = th.optim.Adam(model.parameters(), lr=LR)
    

    while i < MAX_ITERS and not valid: 

        # split training set into regions
        X = data.get_X()

        V_candidate, u = model(X)
        X0, _ = model(data.get_x0())

        
        f1 = system.f1(X, u)
        f2 = system.f2(X, u)


        # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
        L_v_1 = model.get_lie_derivative(V_candidate, X, f1)
        L_v_2 = model.get_lie_derivative(V_candidate, X, f2)


        # Without tuning term
        Lyapunov_risk = (F.relu(-V_candidate) + 1.5*F.relu(L_v_1 + 0.5)+ 1.5*F.relu(L_v_2 + 0.5)).mean() + (X0).pow(2)
        
        
        print(i, "Lyapunov Risk =",Lyapunov_risk.item())
        L.append(Lyapunov_risk.item())
        lyap_risks[out_iters, i] = Lyapunov_risk.item()

        # gradient descent 
        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step() 

        q = model.get_controller_weights()

        # Falsification
        if i % 10 == 0:
            
            # dynamics
            f1, f2 = system.get_fs(falsifier.vars_, q)
            
            #candidate
            V_learn = model.get_candidate(falsifier.vars_)

           
            print('===========Verifying==========')        
            start_ = timeit.default_timer() 

            result = falsifier.check_lyapunov_common(f1, f2, V_learn)

            stop_ = timeit.default_timer() 

 
            if (result): 
                print("V Not a Lyapunov function. Found counterexample: ")
                print(result)

                data.AddCounterexamples(result, 15)

                ce1 = [result[0].mid(), result[1].mid()]

            else:  
                valid = True
                print("V Satisfy conditions!!")
                print(V_learn, " is a common Lyapunov function.")

                ce1 = None



            # plotting
            if False:
                V_plot_1, u_plot = model1(plotter.X_plot)
                V_plot_2 = model2(plotter.X_plot)

                V_plot = system.split_on_data(plotter.X_plot, V_plot_1, V_plot_2)

                f_plot = system.f(plotter.X_plot, u_plot)

                L_plot_1 = model1.get_lie_derivative(V_plot, plotter.X_plot, f_plot)
                L_plot_2 = model2.get_lie_derivative(V_plot, plotter.X_plot, f_plot)

                L_plot = system.split_on_data(plotter.X_plot, L_plot_1, L_plot_2)

                V_plot = V_plot.reshape(plotter.fidelity, plotter.fidelity).cpu().detach().numpy()
                L_plot = L_plot.reshape(plotter.fidelity, plotter.fidelity).cpu().detach().numpy()

                plotter.update_plots(i, V_plot, L_plot, ce1, ce2)


            t += (stop_ - start_)
            print('==============================') 
        i += 1

    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)
    print("Verified time: ", t)
    print(f"Lyapunov Functions found: V={valid}")
    
    out_iters+=1



# V and L plots
# plotter.export_plots()
V_plot, u_plot = model(plotter.X_plot)

f_plot = system.f(plotter.X_plot, u_plot)

L_plot = model.get_lie_derivative(V_plot, plotter.X_plot, f_plot)

V_plot = V_plot.reshape(plotter.fidelity, plotter.fidelity).cpu().detach().numpy()
L_plot = L_plot.reshape(plotter.fidelity, plotter.fidelity).cpu().detach().numpy()

plotter.export_final_lyapunov_function(V_plot)
plotter.export_final_lie_dervative(L_plot)

#ROA => save Lyapunov function
np.save('export/common_lyapunov_function.npy', V_plot)
np.save('export/common_lie_derivative.npy', L_plot)

#Training Points
plotter.export_training_samples(data.get_X().cpu())

# Learning Curve
# export_learning_curve(L, out_iters)

#save NN params
# model.export_model()



