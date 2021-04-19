#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2020                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019-2020                                              ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin

import dolfin_mech as dmech
from .Material_Elastic_Bulk import BulkElasticMaterial

################################################################################

class SkeletonPoroBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            problem,
            parameters):

        self.problem = problem

        assert 'kappa' in parameters
        self.kappa = parameters['kappa']


    def get_dWbulkdJs(self, Phi0, Phi):

        Js = self.problem.kinematics.Je * (1 - Phi)
        dWbulkdJs = self.kappa * (1. / (1. - Phi0) - 1./Js)
        # dWbulkdJs = self.kappa * (1. / (1. - Phi0) - 1./self.problem.kinematics.Js)
        dWbulkdJs = (1 - Phi0) * dWbulkdJs
        # dWbulkdJs = (1 - self.problem.Phi0) * dWbulkdJs

        return dWbulkdJs



###################################################### for mixed formulation ###

    def get_res_term(self, Phi0, Phi, w_U = None, w_Phi0 = None, w_Phi = None):

        assert (w_U is None) or (w_Phi0 is None) or (w_Phi is None)
        assert (w_U is not None) or (w_Phi0 is not None) or (w_Phi is not None)

        dWbulkdJs = self.get_dWbulkdJs(Phi0, Phi)

        if w_U is not None:
            if isinstance(self.problem, dmech.InverseHyperelasticityProblem):
                res_form = dolfin.inner(
                    dWbulkdJs * self.problem.kinematics.I,
                    dolfin.sym(dolfin.grad(self.problem.subsols["U"].dsubtest))) * self.problem.dV

            elif isinstance(self.problem, dmech.HyperelasticityProblem):
                res_form = dolfin.inner(
                    dWbulkdJs * self.problem.kinematics.Je * self.problem.kinematics.Ce_inv,
                    dolfin.derivative(
                            self.problem.kinematics.Et,
                            self.problem.subsols["U"].subfunc,
                            self.problem.subsols["U"].dsubtest)) * self.problem.dV

        elif w_Phi0 is not None:
            res_form = dolfin.inner(
                dWbulkdJs,
                self.problem.subsols["Phi0"].dsubtest) * self.problem.dV

        elif w_Phi is not None:
            res_form = dolfin.inner(
                dWbulkdJs,
                self.problem.subsols["Phi"].dsubtest) * self.problem.dV

        return res_form



########################################## for internal variable formulation ###

    def get_jac_term(self, Phi0, Phi, w_Phi0 = None, w_Phi = None):

        assert (w_Phi0 is None) or (w_Phi is None)
        assert (w_Phi0 is not None) or (w_Phi is not None)

        if w_Phi is not None:

            dWbulkdJs = self.get_dWbulkdJs(Phi0, Phi)
            # Phi = self.problem.get_Phi()
            # if self.problem.w_contact:
            #     dWbulkdJs = self.get_dWbulkdJs(self.problem.Phi0pos, self.problem.Phipos)
            # else:
            #     dWbulkdJs = self.get_dWbulkdJs(self.problem.Phi0, Phi)
            # # dWbulkdJs = self.get_dWbulkdJs(self.problem.Phi0, self.problem.Phi)
            # # dWbulkdJspos = self.get_dWbulkdJs(self.problem.Phi0pos, self.problem.Phipos)

            jac_form = dolfin.inner(
                dolfin.diff(
                    dWbulkdJs * self.problem.kinematics.Je * self.problem.kinematics.Ce_inv,
                    # dWbulkdJspos * self.problem.kinematics.Je * self.problem.kinematics.Ce_inv,
                    self.problem.Phi),
                dolfin.derivative(
                    self.problem.kinematics.Et,
                    self.problem.subsols["U"].subfunc,
                    self.problem.subsols["U"].dsubtest)) * dolfin.inner(
                dolfin.diff(
                    self.problem.get_Phi(),
                    dolfin.variable(self.problem.kinematics.Jt)) * dolfin.diff(
                    self.problem.kinematics.Jt,
                    dolfin.variable(self.problem.kinematics.Ft)),
                dolfin.derivative(
                    self.problem.kinematics.Ft,
                    self.problem.subsols["U"].subfunc,
                    self.problem.subsols["U"].dsubtria)) * self.problem.dV

        elif w_Phi0 is not None:

            dWbulkdJs = self.get_dWbulkdJs(Phi0, Phi)

            jac_form = dolfin.inner(
                dolfin.diff(
                    dWbulkdJs * self.problem.kinematics.I,
                    # dWbulkdJs * self.problem.kinematics.Je * self.problem.kinematics.Ce_inv,
                    # dWbulkdJspos * self.problem.kinematics.Je * self.problem.kinematics.Ce_inv,
                    self.problem.Phi0),
                dolfin.derivative(
                    self.problem.kinematics.Et,
                    self.problem.subsols["U"].subfunc,
                    self.problem.subsols["U"].dsubtest)) * dolfin.inner(
                dolfin.diff(
                    self.problem.get_Phi0(),
                    dolfin.variable(self.problem.kinematics.Jt)) * dolfin.diff(
                    self.problem.kinematics.Jt,
                    dolfin.variable(self.problem.kinematics.Ft)),
                dolfin.derivative(
                    self.problem.kinematics.Ft,
                    self.problem.subsols["U"].subfunc,
                    self.problem.subsols["U"].dsubtria)) * self.problem.dV

        return jac_form
