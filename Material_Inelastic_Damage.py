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
import numpy

import dolfin_cm as dcm
from .Material_Inelastic import InelasticMaterial

################################################################################

class DamageInelasticMaterial(InelasticMaterial):



    def __init__(self,
            problem,
            parameters):

        self.problem = problem

        self.epsilon0 = parameters["epsilon0"]
        self.epsilon1 = parameters["epsilon1"]
        self.gamma    = parameters["gamma"]



###################################################### for mixed formulation ###



    def set_internal_variables_mixed(self):

        pass



    def get_res_term(self,
            dt):

        d_new = ((dolfin.tr(self.problem.kinematics.epsilon) - self.epsilon0)/(self.epsilon1-self.epsilon0))**(self.gamma)
        d_new = dolfin.conditional(dolfin.ge(d_new, 0.), d_new, 0.)
        d_new = dolfin.conditional(dolfin.le(d_new, 1.), d_new, 1.)
        d_new = dolfin.conditional(dolfin.ge(d_new, self.problem.subsols["d"].func_old), d_new, self.problem.subsols["d"].func_old)

        res_form = dolfin.inner(
            self.problem.subsols["d"].subfunc - d_new,
            self.problem.subsols["d"].dsubtest) * self.problem.dV

        return res_form



########################################## for internal variable formulation ###



    def set_internal_variables_internal(self):

        self.d_fs = self.problem.sfoi_fs

        self.d      = dolfin.Function(self.d_fs)
        self.d_test = dolfin.TestFunction(self.d_fs)
        self.d_tria = dolfin.TrialFunction(self.d_fs)

        self.d_new = ((dolfin.tr(self.problem.kinematics.epsilon) - self.epsilon0)/(self.epsilon1-self.epsilon0))**(self.gamma)
        self.d_new = dolfin.conditional(dolfin.ge(self.d_new,     0.), self.d_new, 0.    )
        self.d_new = dolfin.conditional(dolfin.le(self.d_new,     1.), self.d_new, 1.    )
        self.d_new = dolfin.conditional(dolfin.ge(self.d_new, self.d), self.d_new, self.d)

        self.a_expr = dolfin.inner(
            self.d_tria,
            self.d_test) * self.problem.dV
        self.b_expr = dolfin.inner(
            self.d_new,
            self.d_test) * self.problem.dV
        self.local_solver = dolfin.LocalSolver(
            self.a_expr,
            self.b_expr)
        self.local_solver.factorize()



    def update_internal_variables_at_t(self,
            t):

        pass



    def update_internal_variables_after_solve(self,
            dt, t):

        self.local_solver.solve_local_rhs(self.d)
