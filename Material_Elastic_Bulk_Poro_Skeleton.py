#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019                                                   ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin

import dolfin_cm as dcm
from .Material_Elastic_Bulk import BulkElasticMaterial

################################################################################

class SkeletonPoroBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            problem,
            kappa):

        self.problem = problem

        self.kappa = kappa



    def get_dWbulkdJs(self, Phi0):

        dWbulkdJs = self.kappa * (1. / (1. - Phi0) - 1./self.problem.kinematics.Js)
        dWbulkdJs = (1 - self.problem.Phi0) * dWbulkdJs

        return dWbulkdJs



###################################################### for mixed formulation ###

    def get_res_term(self, Phi0, w_U = None, w_Phi = None):

        assert (w_U is None) or (w_Phi is None)

        dWbulkdJs = self.get_dWbulkdJs(Phi0)

        if w_U is not None:
            res_form = dolfin.inner(
                dWbulkdJs * self.problem.kinematics.Je * self.problem.kinematics.Ce_inv,
                dolfin.derivative(
                        self.problem.kinematics.Et,
                        self.problem.subsols["U"].subfunc,
                        self.problem.subsols["U"].dsubtest)) * self.problem.dV
        elif w_Phi is not None:
            res_form = dolfin.inner(
                dWbulkdJs,
                self.problem.subsols["Phi"].dsubtest) * self.problem.dV

        return res_form



########################################## for internal variable formulation ###

    def get_jac_term(self):

        dWbulkdJs = self.get_dWbulkdJs(self.problem.Phi0)
        dWbulkdJspos = self.get_dWbulkdJs(self.problem.Phi0pos)
        Phi = self.problem.get_Phi()

        jac_form = dolfin.inner(
            dolfin.diff(
                dWbulkdJspos * self.problem.kinematics.Je * self.problem.kinematics.Ce_inv,
                self.problem.Phi),
            dolfin.derivative(
                self.problem.kinematics.Et,
                self.problem.subsols["U"].subfunc,
                self.problem.subsols["U"].dsubtest)) * dolfin.inner(
            dolfin.diff(
                Phi,
                dolfin.variable(self.problem.kinematics.Jt)) * dolfin.diff(
                self.problem.kinematics.Jt,
                dolfin.variable(self.problem.kinematics.Ft)),
            dolfin.derivative(
                self.problem.kinematics.Ft,
                self.problem.subsols["U"].subfunc,
                self.problem.subsols["U"].dsubtria)) * self.problem.dV

        return jac_form
