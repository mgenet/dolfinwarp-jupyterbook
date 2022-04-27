#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech

################################################################################

class TimeVaryingConstant():



    def __init__(self,
            val=None, val_ini=None, val_fin=None):

        if  (val     is not None)\
        and (val_ini is     None)\
        and (val_fin is     None):
            val_ini = val
            val_fin = val
        elif (val     is     None)\
         and (val_ini is not None)\
         and (val_fin is not None):
            pass
        else:
            assert (0), "Must provide val or val_ini & val_fin. Aborting."

        assert (type(val_ini) in (int, float, list, numpy.ndarray))
        if (type(val_ini) in (int, float)):
            assert (type(val_fin) in (int, float))
            self.val_ini = numpy.array([val_ini])
            self.val_fin = numpy.array([val_fin])
            self.val_cur = numpy.array([val_ini])
            self.val_old = numpy.array([val_ini])
            self.set_value = self.set_value_sca
        elif (type(val_ini) in (list, numpy.ndarray)):
            assert (type(val_fin) in (list, numpy.ndarray))
            self.val_ini = numpy.array(val_ini)
            self.val_fin = numpy.array(val_fin)
            self.val_cur = numpy.array(val_ini)
            self.val_old = numpy.array(val_ini)
            self.set_value = self.set_value_vec
        self.val = dolfin.Constant(val_ini)



    def set_value_sca(self,
            val):

        if   (type(val) in (int, float)):
            self.val.assign(dolfin.Constant(val))
        elif (type(val) in (list, numpy.ndarray)):
            self.val.assign(dolfin.Constant(val[0]))



    def set_value_vec(self,
            val):

        self.val.assign(dolfin.Constant(val))



    def set_value_at_t_step(self,
            t_step):

        self.set_value(self.val_ini * (1. - t_step) + self.val_fin * t_step)



    def set_dvalue_at_t_step(self,
            t_step):

        self.val_old[:] = self.val_cur[:]
        self.val_cur[:] = self.val_ini * (1. - t_step) + self.val_fin * t_step
        self.set_value(self.val_cur - self.val_old)



    def restore_old_value(self):

        self.val_cur[:] = self.val_old[:]



    def homogenize(self):

        self.set_value(0*self.val_ini)
