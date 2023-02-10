import dolfin

import dolfin_mech as dmech
from .Operator import Operator

import numpy
# ################################################################################

class DeformedTotalVolumeOperator(Operator):

    def __init__(self,
            v,
            v_test,
            U_bar,
            V0,
            Vs0, 
            measure):

        self.measure = measure

        self.U_bar = U_bar
        dim = self.U_bar.ufl_shape[0]
        self.F_bar = dolfin.Identity(dim) + self.U_bar
        self.J_bar = dolfin.det(self.F_bar)
        
        self.res_form = ((v - self.J_bar*V0) * v_test)/Vs0 * self.measure
