import dolfin

import dolfin_mech as dmech
from .Operator import Operator

# ################################################################################

class x0DirectOperator(Operator):

    def __init__(self,
            x0,
            x0_test,
            X,
            U,
            phis0,
            V0, 
            rho0,
            measure):

        self.V0 = dolfin.Constant(V0)
        self.measure = measure

        self.res_form =  dolfin.inner(( (X+U)*phis0/rho0 - x0/self.V0), x0_test) * self.measure
