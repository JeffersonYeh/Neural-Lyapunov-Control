from dreal import *


class Falsifier:

    def __init__(self, ball_lb = 0.5, ball_ub=6, epsilon=0):
        
        self.x1 = Variable("x1")
        self.x2 = Variable("x2")
        self.vars_ = [self.x1, self.x2]


       
        self.config = Config()
        self.config.use_polytope_in_forall = True
        self.config.use_local_optimization = True
        self.config.precision = 1e-2

        self.epsilon = epsilon
        # Checking candidate V within a ball around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub)
        self.ball_lb = ball_lb
        self.ball_ub = ball_ub


    def check_lyapunov(self, f, V):
        # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
        # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub). 
        # If it return unsat, then there is no state violating the conditions. 

        ball= Expression(0)
        lie_derivative_of_V = Expression(0)
        
        for i in range(len(self.vars_)):
            ball += self.vars_[i] * self.vars_[i]
            lie_derivative_of_V += f[i]*V.Differentiate(self.vars_[i])  

        ball_in_bound = logical_and(self.ball_lb * self.ball_lb <= ball, ball <= self.ball_ub * self.ball_ub)
        
        # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     
        condition = logical_and(logical_imply(ball_in_bound, V >= 0),
                                logical_imply(ball_in_bound, lie_derivative_of_V <= self.epsilon)
                               )
        
        return CheckSatisfiability(logical_not(condition), self.config)



    def check_lyapunov_select(self, f, V, switch_condition):    
    
        ball= Expression(0)
        lie_derivative_of_V = Expression(0)
        
        for i in range(len(self.vars_)):
            ball += self.vars_[i] * self.vars_[i]
            lie_derivative_of_V += f[i]*V.Differentiate(self.vars_[i])

        ball_in_bound = logical_and(self.ball_lb * self.ball_lb <= ball, ball <= self.ball_ub * self.ball_ub, switch_condition)
        
        # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     
        condition = logical_and(logical_imply(ball_in_bound, V >= 0),
                                logical_imply(ball_in_bound, lie_derivative_of_V <= self.epsilon),
                                )
        
        return CheckSatisfiability(logical_not(condition), self.config)
    

    def check_lyapunov_common(self, f1, f2, V): 
        ball= Expression(0)
        lie_derivative_of_V_1 = Expression(0)
        lie_derivative_of_V_2 = Expression(0)
        
        for i in range(len(self.vars_)):
            ball += self.vars_[i] * self.vars_[i]
            lie_derivative_of_V_1 += f1[i]*V.Differentiate(self.vars_[i])  
            lie_derivative_of_V_2 += f2[i]*V.Differentiate(self.vars_[i])  

        ball_in_bound = logical_and(self.ball_lb * self.ball_lb <= ball, ball <= self.ball_ub * self.ball_ub)
        
        # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     
        condition = logical_and(logical_imply(ball_in_bound, V >= 0),
                                logical_imply(ball_in_bound, lie_derivative_of_V_1 <= self.epsilon),
                                logical_imply(ball_in_bound, lie_derivative_of_V_2 <= self.epsilon)
                               )
        
        return CheckSatisfiability(logical_not(condition), self.config)