
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
MAX_ITERS = 1000


##

# system = InvertedPendulum(device)
system = LinearSwitchStable(device)

data = DomainSampler(N, D_in, device)
falsifier = Falsifier()
plotter = Plotter(device, fidelity=100)

valid_1 = False
valid_2 = False



lyap_risks = np.zeros((TRIES, MAX_ITERS))

while out_iters < TRIES and (not valid_1 or not valid_2): 
    start = timeit.default_timer()

    if not valid_1:
        model1 = Learner(D_in, H1, D_out, controller=True, lqr=system.get_lqr()).to(device)
        optimizer1 = th.optim.RMSprop(model1.parameters(), lr=LR, alpha=0.99, eps=1e-8, weight_decay=1e-4)
        L1 = []

    if not valid_2:
        model2 = Learner(D_in, H1, D_out, controller=False).to(device)
        optimizer2 = th.optim.RMSprop(model2.parameters(), lr=LR, alpha=0.99, eps=1e-8, weight_decay=1e-4)
        L2 = []


    plotter.reset_plots()

    # L = []
    i = 0 
    t = 0
    
    # optimizer = th.optim.Adam(model.parameters(), lr=LR)
    

    while i < MAX_ITERS and (not valid_1 or not valid_2): 

        # split training set into regions
        X1, X2 = system.split_samples(data.get_X())

        V_candidate_1, u_1 = model1(X1)
        X0_1, _ = model1(data.get_x0())

        V_candidate_2 = model2(X2)
        X0_2 = model2(data.get_x0())

        _, u_2 = model1(X2)
        
        f_1 = system.f(X1, u_1)
        f_2 = system.f(X2, u_2)


        # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
        L_v_1 = model1.get_lie_derivative(V_candidate_1, X1, f_1)
        L_v_2 = model2.get_lie_derivative(V_candidate_2, X2, f_2)


        # Without tuning term
        Lyapunov_risk_1 = (F.relu(-V_candidate_1) + 2.5*F.relu(L_v_1 + 0.5)).mean() + (X0_1).pow(2)
        Lyapunov_risk_2 = (F.relu(-V_candidate_2) + 2.5*F.relu(L_v_2 + 0.5)).mean() + (X0_2).pow(2)
        
        
        print(i, "Lyapunov Risk 1 =",Lyapunov_risk_1.item(), "Lyapunov Risk 2 =",Lyapunov_risk_2.item())
        L1.append(Lyapunov_risk_1.item())
        L2.append(Lyapunov_risk_2.item())
        lyap_risks[out_iters, i] = Lyapunov_risk_1.item() + Lyapunov_risk_2.item()

        # gradient descent 
        if not valid_1:
            optimizer1.zero_grad()
            Lyapunov_risk_1.backward()
            optimizer1.step() 

        if not valid_2:
            optimizer2.zero_grad()
            Lyapunov_risk_2.backward()
            optimizer2.step() 

        q = model1.get_controller_weights()

        # Falsification
        if i % 10 == 0:
            
            # dynamics
            f1, f2 = system.get_fs(falsifier.vars_, q)
            
            #candidate
            V_learn_1 = model1.get_candidate(falsifier.vars_)
            V_learn_2 = model2.get_candidate(falsifier.vars_)

           


            print('===========Verifying==========')        
            start_ = timeit.default_timer() 

            result_1 = falsifier.check_lyapunov_select(f1, V_learn_1, system.get_switch_condition(falsifier.vars_))
            result_2 = falsifier.check_lyapunov_select(f2, V_learn_2, logical_not(system.get_switch_condition(falsifier.vars_)))

            stop_ = timeit.default_timer() 

            if not valid_1:
                if (result_1): 
                    print("V1 Not a Lyapunov function. Found counterexample: ")
                    print(result_1)

                    data.AddCounterexamples(result_1, 15)

                    ce1 = [result_1[0].mid(), result_1[1].mid()]

                else:  
                    valid_1 = True
                    print("V1 Satisfy conditions!!")
                    print(V_learn_1, "on x_1 <= 0 is a Lyapunov function.")

                    ce1 = None


            if not valid_2:
                if (result_2): 
                    print("V2 Not a Lyapunov function. Found counterexample: ")
                    print(result_2)

                    data.AddCounterexamples(result_2, 15)
                    ce2 = [result_2[0].mid(), result_2[1].mid()]

                else:  
                    valid_2 = True
                    print("V2 Satisfy conditions!!")
                    print(V_learn_2, " on x_1 > 0 is a Lyapunov function.")

                    ce2 = None


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
    print(f"Lyapunov Functions found: V1={valid_1}, V2={valid_2}")
    
    out_iters+=1



# V and L plots
# plotter.export_plots()
V_plot_1, u_plot = model1(plotter.X_plot)
V_plot_2 = model2(plotter.X_plot)

V_plot = system.split_on_data(plotter.X_plot, V_plot_1, V_plot_2)

f_plot = system.f(plotter.X_plot, u_plot)

L_plot_1 = model1.get_lie_derivative(V_plot, plotter.X_plot, f_plot)
L_plot_2 = model2.get_lie_derivative(V_plot, plotter.X_plot, f_plot)

L_plot = system.split_on_data(plotter.X_plot, L_plot_1, L_plot_2)

V_plot = V_plot.reshape(plotter.fidelity, plotter.fidelity).cpu().detach().numpy()
L_plot = L_plot.reshape(plotter.fidelity, plotter.fidelity).cpu().detach().numpy()

plotter.export_final_lyapunov_function(V_plot)
plotter.export_final_lie_dervative(L_plot)

#ROA => save Lyapunov function
np.save('export/lyapunov_function.npy', V_plot)
np.save('export/lie_derivative.npy', L_plot)

#Training Points
plotter.export_training_samples(data.get_X().cpu())

# Learning Curve
export_learning_curve(L1, L2, out_iters)

#save NN params
model1.export_model()
model2.export_model()



