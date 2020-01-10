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
from .Material_Inelastic_Growth import GrowthInelasticMaterial

################################################################################



class TimeDrivenGrowthInelasticMaterial(GrowthInelasticMaterial):



    def __init__(self,
            parameters):

        self.taug = parameters["taug"]



    def get_thetag_dot(self):

        return 1./self.taug



    def get_thetag_at_t(self,
            t):

        return self.get_thetag_dot()*t



    def get_Fg(self,
            thetag):

        return (1+thetag) * numpy.eye(self.dim)



    def get_Fg_expr(self,
            thetag):

        return (1+thetag) * dolfin.Identity(self.dim)



###################################################### for mixed formulation ###



    def set_internal_variables_mixed(self,
            problem):

        self.dim = problem.dim

        self.Fg     = self.get_Fg_expr(thetag=problem.subsols["thetag"].subfunc )
        self.Fg_old = self.get_Fg_expr(thetag=problem.subsols["thetag"].func_old)

        problem.add_foi(
            expr=self.Fg,
            fs=problem.mfoi_fs,
            name="Fg")



    def get_res_term(self,
            problem,
            dt):

        thetag_dot = dolfin.Constant(self.get_thetag_dot())

        thetag_new = problem.subsols["thetag"].func_old + thetag_dot * dolfin.Constant(dt)

        res_form = dolfin.inner(problem.subsols["thetag"].subfunc - thetag_new, problem.subsols["thetag"].dsubtest) * problem.dV

        return res_form



########################################## for internal variable formulation ###



    def set_internal_variables_internal(self,
            problem):

        self.dim = problem.dim

        self.thetag     = dolfin.Constant(self.get_thetag_at_t(t=0.))
        self.thetag_old = dolfin.Constant(self.get_thetag_at_t(t=0.))

        self.Fg     = dolfin.Constant(self.get_Fg(thetag=self.thetag    ))
        self.Fg_old = dolfin.Constant(self.get_Fg(thetag=self.thetag_old))

        problem.add_foi(
            expr=self.thetag,
            fs=problem.sfoi_fs,
            name="thetag")
        problem.add_foi(
            expr=self.Fg,
            fs=problem.mfoi_fs,
            name="Fg")



    def update_internal_variables_at_t(self,
            problem,
            t):

        self.thetag_old.assign(self.thetag)
        self.Fg_old.assign(self.Fg)
        self.thetag.assign(self.get_thetag_at_t(t=t))
        self.Fg.assign(dolfin.Constant(self.get_Fg(thetag=self.thetag.values()[0])))



    def update_internal_variables_after_solve(self,
            problem,
            dt, t):

        pass
