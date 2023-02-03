import dolfin

import dolfin_mech as dmech
from .Operator import Operator


# # ################################################################################

class DeformedMatrixOperator1(Operator):

    def __init__(self,
            M1_d,
            M1_test,
            kinematics,
            N,
            S0, 
            measure):

        self.S0 = dolfin.Constant(S0)
        self.measure = measure

        n = dolfin.dot(N,dolfin.inv(kinematics.F))
        intermediaire = dolfin.outer(n, n)/(1/2*dolfin.inner(n, n))**(1/2)
        
        # print("type M1", type(intermediaire))
        
        self.res_form =  dolfin.inner((intermediaire - M1_d/self.S0), M1_test) * kinematics.J * self.measure

# # ################################################################################

class DeformedMatrixOperator2(Operator):

    def __init__(self,
            X,
            U,
            x0,
            M2_d,
            M2_test,
            kinematics,
            N,
            S0, 
            measure):
        
        
        x = X + U
        x_tilde = x-x0
        # print("x_tilde=", x_tilde)
        n = dolfin.dot(N,dolfin.inv(kinematics.F))

        # intermediaire = dolfin.outer(dolfin.dot(N,dolfin.inv(kinematics.F)), dolfin.dot(N,dolfin.inv(kinematics.F)))*2/dolfin.inner(dolfin.dot(N,dolfin.inv(kinematics.F)),dolfin.dot(N,dolfin.inv(kinematics.F)))**(1/2)

        intermediaire = dolfin.outer(n, dolfin.cross(x_tilde, n))/(1/2*dolfin.inner(n,n))**(1/2)
        # print("type M2", type(intermediaire))
        
        self.S0 = dolfin.Constant(S0)
        self.measure = measure
        self.res_form = dolfin.inner((intermediaire - M2_d/self.S0), M2_test) * kinematics.J * self.measure

# # ################################################################################

class DeformedMatrixOperator3(Operator):

    def __init__(self,
            X,
            U,
            x0,
            M3_d,
            M3_test,
            kinematics,
            N,
            S0, 
            measure):
        
        x = X + U
        x_tilde = x-x0

        xtilde_matrix = [[0, -x_tilde[2], x_tilde[1]], [x_tilde[2], 0, -x_tilde[0]], [-x_tilde[1], x_tilde[0],0]]
        n = dolfin.dot(N,dolfin.inv(kinematics.F))
        n_outer = dolfin.outer(n,n)
        intermediaire = dolfin.as_matrix(xtilde_matrix)*n_outer/(1/2*dolfin.inner(n, n))**(1/2)

        # print("type M3  ", type(intermediaire))

        
        self.S0 = dolfin.Constant(S0)
        self.measure = measure
        self.res_form = dolfin.inner((intermediaire - M3_d/self.S0), M3_test) * kinematics.J * self.measure


# # ################################################################################

class DeformedMatrixOperator4(Operator):

    def __init__(self,
            X,
            U,
            x0,
            M4_d,
            M4_test,
            kinematics,
            N,
            S0, 
            measure):
        
        x = X + U
        x_tilde = x-x0


        xtilde_matrix = [[0, -x_tilde[2], x_tilde[1]], [x_tilde[2], 0, -x_tilde[0]], [-x_tilde[1], x_tilde[0],0]]
        n = dolfin.dot(N,dolfin.inv(kinematics.F))
        
        intermediaire = dolfin.as_matrix(xtilde_matrix)*dolfin.outer(n,dolfin.cross(x_tilde, n))/(1/2*dolfin.inner(n, n))**(1/2)
        
        # print("type M4", type(intermediaire))

        self.S0 = dolfin.Constant(S0)
        self.measure = measure
        self.res_form = dolfin.inner((intermediaire - M4_d/self.S0), M4_test) * kinematics.J * self.measure
