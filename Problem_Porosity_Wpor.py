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
import numpy

import dolfin_mech as dmech
from .Problem_Hyperelasticity import HyperelasticityProblem

################################################################################

class PoroWporProblem(HyperelasticityProblem):



    def __init__(self,
            w_incompressibility=False):

        HyperelasticityProblem.__init__(self,w_incompressibility)



    def set_kinematics(self):

        HyperelasticityProblem.set_kinematics(self)
        self.kinematics.Phi = 1 - (1 - self.porosity0) / self.kinematics.Je
        self.kinematics.Js = self.kinematics.Je * (1-self.kinematics.Phi)



    def set_variational_formulation(self,
            normal_penalties=[],
            directional_penalties=[],
            surface_tensions=[],
            surface0_loadings=[],
            pressure0_loadings=[],
            volume0_loadings=[],
            surface_loadings=[],
            pressure_loadings=[],
            volume_loadings=[],
            dt=None):

        self.Pi = sum([subdomain.Psi * self.dV(subdomain.id) for subdomain in self.subdomains])
        # print (self.Pi)

        self.res_form = dolfin.derivative(
            self.Pi,
            self.sol_func,
            self.dsol_test);

        for loading in pressure_loadings:
            T = dolfin.dot(
               -loading.val * self.mesh_normals,
                dolfin.inv(self.kinematics.Ft))
            self.res_form -= self.kinematics.Jt * dolfin.inner(
                T,
                self.subsols["U"].dsubtest) * loading.measure

        dWpordJ = - self.eta_por / (self.kinematics.Je - self.kinematics.Js)
        self.res_form += dolfin.inner(
            dWpordJ * self.kinematics.Je * self.kinematics.Ce_inv,
            dolfin.derivative(
                    self.kinematics.Et,
                    self.subsols["U"].subfunc,
                    self.subsols["U"].dsubtest)) * self.dV

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)



    def add_Phydro_qois(self):

        n_subdomains = 0
        for subdomain in self.subdomains:
            n_subdomains += 1
        if n_subdomains == 1:
            basename = "Phydro_"
            P = -1./3. * dolfin.tr(self.subdomains[0].sigma)

        self.add_qoi(
            name=basename,
            expr=P * self.dV)



    def add_dPsiBulkdJs_qois(self):

        n_subdomains = 0
        for subdomain in self.subdomains:
            n_subdomains += 1
        if n_subdomains == 1:
            basename = "dPsiBulkdJs_"
            kappa = 10**(9)
            deriv = kappa * (1 / (1 - self.porosity0) - 1 / self.kinematics.Js)

        self.add_qoi(
            name=basename,
            expr=deriv * self.dV)



    def add_dPsiPordJ_qois(self):

        n_subdomains = 0
        for subdomain in self.subdomains:
            n_subdomains += 1
        if n_subdomains == 1:
            basename = "dPsiPordJ_"
            deriv = - self.eta_por / (self.kinematics.Je - self.kinematics.Js)

        self.add_qoi(
            name=basename,
            expr=deriv * self.dV)


    def add_Phi_qois(self):

        basename = "PHI_"
        Phi = self.kinematics.Phi

        self.add_qoi(
            name=basename,
            expr=Phi * self.dV)


    def add_Js_qois(self):

        basename = "Js_"

        self.add_qoi(
            name=basename,
            expr=self.kinematics.Js * self.dV)
