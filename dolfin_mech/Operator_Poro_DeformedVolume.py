import dolfin

import dolfin_mech as dmech
from .Operator import Operator

# ################################################################################

class DeformedVolumeOperator(Operator):

    def __init__(self,
            v,
            v_test,
            J,
            V0, 
            measure):

        self.V0 = dolfin.Constant(V0)
        self.measure = measure
        
        self.res_form =  ((J - v/self.V0) * v_test) * self.measure
