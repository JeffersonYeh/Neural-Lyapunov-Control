# -*- coding: utf-8 -*-
from dreal import *
import torch 
import numpy as np
import random


def CheckLyapunov(x, f, V, ball_lb, ball_ub, config, epsilon):    
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
    # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub). 
    # If it return unsat, then there is no state violating the conditions. 
    
    ball= Expression(0)
    lie_derivative_of_V = Expression(0)
    
    for i in range(len(x)):
        ball += x[i]*x[i]
        lie_derivative_of_V += f[i]*V.Differentiate(x[i])  
    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    
    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     
    condition = logical_and(logical_imply(ball_in_bound, V >= 0),
                           logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon))
    return CheckSatisfiability(logical_not(condition),config)

def CheckLyapunovRelaxed(x, f1, f2, switch_condition, V, ball_lb, ball_ub, config, epsilon):
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
    # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub). 
    # If it return unsat, then there is no state violating the conditions. 
    
    ball= Expression(0)
    lie_derivative_of_V_1 = Expression(0)
    lie_derivative_of_V_2 = Expression(0)
    
    for i in range(len(x)):
        ball += x[i]*x[i]
        lie_derivative_of_V_1 += f1[i]*V.Differentiate(x[i])
        lie_derivative_of_V_2 += f2[i]*V.Differentiate(x[i])

    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    ball_in_bound_1 = logical_and(ball_in_bound, switch_condition)
    ball_in_bound_2 = logical_and(ball_in_bound, logical_not(switch_condition))

    
    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     

    condition = logical_and(logical_imply(ball_in_bound, V >= 0),
                            logical_imply(ball_in_bound_1, lie_derivative_of_V_1 <= epsilon),
                            logical_imply(ball_in_bound_2, lie_derivative_of_V_2 <= epsilon),
                            )
    
    return CheckSatisfiability(logical_not(condition),config)


def CheckLyapunovMultiple(x, f1, f2, switch_condition, V_1, V_2, ball_lb, ball_ub, config, epsilon):
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V1 and V2
    # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub). 
    # If it return unsat, then there is no state violating the conditions. 
    
    ball= Expression(0)
    lie_derivative_of_V_1 = Expression(0)
    lie_derivative_of_V_2 = Expression(0)
    
    for i in range(len(x)):
        ball += x[i]*x[i]
        lie_derivative_of_V_1 += f1[i]*V_1.Differentiate(x[i])
        lie_derivative_of_V_2 += f2[i]*V_2.Differentiate(x[i])

    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    ball_in_bound_1 = logical_and(ball_in_bound, switch_condition)
    ball_in_bound_2 = logical_and(ball_in_bound, logical_not(switch_condition))

    
    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     

    condition = logical_and(logical_imply(ball_in_bound_1, V_1 >= 0),
                            logical_imply(ball_in_bound_2, V_2 >= 0),
                            logical_imply(ball_in_bound_1, lie_derivative_of_V_1 <= epsilon),
                            logical_imply(ball_in_bound_2, lie_derivative_of_V_2 <= epsilon),
                            )
    
    return CheckSatisfiability(logical_not(condition),config)

def CheckLyapunovMultipleSelect(x, f, switch_condition, V, ball_lb, ball_ub, config, epsilon):
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V1 and V2
    # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub). 
    # If it return unsat, then there is no state violating the conditions. 
    
    ball= Expression(0)
    lie_derivative_of_V = Expression(0)
    
    for i in range(len(x)):
        ball += x[i]*x[i]
        lie_derivative_of_V += f[i]*V.Differentiate(x[i])

    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub, switch_condition)
    
    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     

    condition = logical_and(logical_imply(ball_in_bound, V >= 0),
                            logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon),
                            )
    
    return CheckSatisfiability(logical_not(condition),config)




def AddCounterexamples(x,CE,N, replace=False): 

    # if x.shape[0] > 2000 => dont expand set

    if x.shape[0] >= 500 and replace:
        idx = torch.randperm(x.shape[0], device=x.device)[:-N]
        x = x[idx]

    # Adding CE back to sample set
    c = []
    nearby= []
    for i in range(CE.size()):
        c.append(CE[i].mid())
        lb = CE[i].lb()
        ub = CE[i].ub()
        nearby_ = np.random.uniform(lb,ub,N)
        nearby.append(nearby_)

    for i in range(N):
        n_pt = []
        for j in range(x.shape[1]):
            n_pt.append(nearby[j][i])             
        x = torch.cat((x, torch.tensor([n_pt], device=x.device)), 0)
    return x


def SampleCounterexamples(CE,N, device): 

    samples = []

    # Adding CE back to sample set
    c = []
    nearby= []
    for i in range(CE.size()):
        c.append(CE[i].mid())
        lb = CE[i].lb()
        ub = CE[i].ub()
        nearby_ = np.random.uniform(lb,ub,N)
        nearby.append(nearby_)

    for i in range(N):
        n_pt = []
        for j in range(CE.size()):
            n_pt.append(nearby[j][i])    

        samples.append(n_pt)

    return torch.tensor(samples, device=device)
  
def dtanh(s):
    # Derivative of activation
    return 1.0 - s**2

def Tune(x):
    # Circle function values
    y = []
    for r in range(0,len(x)):
        v = 0 
        for j in range(x.shape[1]):
            v += x[r][j]**2
        f = [torch.sqrt(v)]
        y.append(f)
    y = torch.tensor(y, device=x.device)
    return y