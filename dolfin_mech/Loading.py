#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2020                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin

import dolfin_mech as dmech

################################################################################

class Loading():



    def __init__(self,
            measure,
            val=None,
            val_ini=None,
            val_fin=None,
            N=None):

        self.measure = measure

        if (val is not None) and (val_ini is None) and (val_fin is None):
            self.tv_val = dmech.TimeVaryingConstant(
                val_ini=val,
                val_fin=val)
        elif (val is None) and (val_ini is not None) and (val_fin is not None):
            self.tv_val = dmech.TimeVaryingConstant(
                val_ini=val_ini,
                val_fin=val_fin)
        self.val = self.tv_val.val

        if (N is not None):
            self.N = dolfin.Constant(N)



    def set_value(self,
            val):

        self.tv_val.set_value(val)



    def set_value_at_t_step(self,
            t_step):

        self.tv_val.set_value_at_t_step(t_step)
