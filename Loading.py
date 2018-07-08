#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018                                            ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_cm as dcm

################################################################################

class Loading():



    def __init__(self,
            measure,
            val=None,
            val_ini=None,
            val_fin=None):

        self.measure = measure

        if (val is not None) and (val_ini is None) and (val_fin is None):
            self.tv_val = dcm.TimeVaryingConstant(
                val_ini=val,
                val_fin=val)
        elif (val is None) and (val_ini is not None) and (val_fin is not None):
            self.tv_val = dcm.TimeVaryingConstant(
                val_ini=val_ini,
                val_fin=val_fin)
        self.val = self.tv_val.val



    def set_value(self,
            val):

        self.tv_val.set_value(val)



    def set_value_at_t_step(self,
            t_step):

        self.tv_val.set_value_at_t_step(t_step)
