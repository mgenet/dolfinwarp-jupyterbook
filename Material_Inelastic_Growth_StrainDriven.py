#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
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



class StrainDrivenGrowthInelasticMaterial(GrowthInelasticMaterial):



    def __init__(self,
            parameters):

        self.thetag_max = parameters["thetag_max"]
        self.Ee_thr = parameters["Ee_thr"]
        self.taug = parameters["taug"]



    def get_Fg(self,
            thetag):

        return (1+thetag) * self.I



###################################################### for mixed formulation ###



    def set_internal_variables_mixed(self,
            problem):

        self.I = dolfin.Identity(problem.dim)

        self.thetag     = problem.subsols["thetag"].subfunc
        self.thetag_old = problem.subsols["thetag"].func_old

        self.Fg     = self.get_Fg(thetag=self.thetag    )
        self.Fg_old = self.get_Fg(thetag=self.thetag_old)

        problem.add_foi(
            expr=self.Fg,
            fs=problem.mfoi_fs,
            name="Fg")



    def get_res_term(self,
            problem,
            dt):

        Ee_mid_lp = (dolfin.tr(problem.kinematics.Ee_mid) + dolfin.sqrt(dolfin.tr(problem.kinematics.Ee_mid)**2 - 4*dolfin.det(problem.kinematics.Ee_mid)))/2
        Ee_mid_lm = (dolfin.tr(problem.kinematics.Ee_mid) - dolfin.sqrt(dolfin.tr(problem.kinematics.Ee_mid)**2 - 4*dolfin.det(problem.kinematics.Ee_mid)))/2
        Ee_mid_lp_pos = dolfin.conditional(dolfin.ge(Ee_mid_lp, 0.), Ee_mid_lp, 0.)
        Ee_mid_lm_pos = dolfin.conditional(dolfin.ge(Ee_mid_lm, 0.), Ee_mid_lm, 0.)
        deltaE = (dolfin.sqrt(Ee_mid_lp_pos**2 + Ee_mid_lm_pos**2) - dolfin.Constant(self.Ee_thr)) / dolfin.Constant(self.Ee_thr)

        #deltaE = (dolfin.sqrt(dolfin.inner(
            #problem.kinematics.Ee_mid,
            #problem.kinematics.Ee_mid)) - dolfin.Constant(self.Ee_thr)) / dolfin.Constant(self.Ee_thr)

        deltaE_pos = dolfin.conditional(dolfin.ge(deltaE, 0.), deltaE, 0.)

        thetag_mid = (problem.subsols["thetag"].func_old + problem.subsols["thetag"].subfunc)/2

        delatthetag = ((dolfin.Constant(self.thetag_max) - thetag_mid) / dolfin.Constant(self.thetag_max))

        delatthetag_pos = dolfin.conditional(dolfin.ge(delatthetag, 0.), delatthetag, 0.)

        thetag_dot = delatthetag_pos**2 * deltaE_pos**2 / dolfin.Constant(self.taug)

        thetag_new = problem.subsols["thetag"].func_old + thetag_dot * dolfin.Constant(dt)

        res_form = dolfin.inner(
            problem.subsols["thetag"].subfunc - thetag_new,
            problem.subsols["thetag"].dsubtest) * problem.dV

        return res_form



########################################## for internal variable formulation ###



    def set_internal_variables_internal(self,
            problem):

        assert (0)



    def update_internal_variables_at_t(self,
            problem,
            t):

        assert (0)



    def update_internal_variables_after_solve(self,
            problem,
            dt, t):

        assert (0)
