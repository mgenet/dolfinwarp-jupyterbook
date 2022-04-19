#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator
from .Problem import Problem

################################################################################

class MacroscopicStrainComponentPenaltyOperator(Operator):

    def __init__(self,
            epsilon_bar,
            epsilon_bar_test,
            i,
            j,
            val,
            measure,
            pen=None,
            pen_ini=None,
            pen_fin=None):

        self.measure = measure
        self.tv_pen = dmech.TimeVaryingConstant(
            val=pen, val_ini=pen_ini, val_fin=pen_fin)
        pen = self.tv_pen.val

        Pi = (pen/2) * (epsilon_bar[i,j] - val)**2
        self.res_form = dolfin.derivative(Pi, epsilon_bar[i,j], epsilon_bar_test[i,j]) * self.measure
        # epsilon_bar = dolfin.variable(epsilon_bar)
        # Pi = (pen/2) * dolfin.inner(epsilon_bar - val, epsilon_bar - val)
        # self.res_form = dolfin.derivative(Pi, epsilon_bar, epsilon_bar_test) * self.measure
        


    def set_value_at_t_step(self,
            t_step):

        self.tv_pen.set_value_at_t_step(t_step)
