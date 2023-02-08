import dolfin

import dolfin_mech as dmech
from .Operator import Operator

# ################################################################################

class DeformedSolidVolumeOperator(Operator):

    def __init__(self,
            vs,
            vs_test,
            J,
            Vs0, 
            measure):

        self.Vs0 = dolfin.Constant(Vs0)
        self.measure = measure
        
        self.res_form = ((J - vs/self.Vs0) * vs_test) * self.measure
