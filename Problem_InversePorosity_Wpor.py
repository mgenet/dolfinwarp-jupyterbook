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
from .Problem_InverseHyperelasticity import InverseHyperelasticityProblem

################################################################################

class InversePoroWporProblem(InverseHyperelasticityProblem):



    def __init__(self,
            eta,
            kappa):

        InverseHyperelasticityProblem.__init__(self,w_incompressibility=False)
        self.eta = eta
        self.kappa = kappa



    def set_kinematics(self):

        InverseHyperelasticityProblem.set_kinematics(self)

        self.kinematics.Phi = 1 - (1 - self.porosity0) * self.kinematics.Je
        self.kinematics.Js = (1-self.kinematics.Phi) / self.kinematics.Je

        # work
        # self.kinematics.Phi = 1 - (1 - self.porosity0) * self.kinematics.Je**2
        # self.kinematics.Js = (1-self.kinematics.Phi) / self.kinematics.Je

        #
        # Js = (1-self.porosity0)* self.kinematics.Je/2 * (1 + 1/((1-self.porosity0)* self.kinematics.Je**2) + self.eta/self.kappa - ((1 + 1/((1-self.porosity0)* self.kinematics.Je**2) + self.eta/self.kappa)**2 - 4 / ((1-self.porosity0)* self.kinematics.Je**2))**(1./2.))
        # self.kinematics.Phi = 1 - Js / self.kinematics.Je
        # self.kinematics.Js = (1-self.kinematics.Phi) / self.kinematics.Je

    def set_materials(self,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None,
            subdomain_id=None):

        InverseHyperelasticityProblem.set_materials(self,
                elastic_behavior=elastic_behavior,
                elastic_behavior_dev=elastic_behavior_dev,
                elastic_behavior_bulk=elastic_behavior_bulk,
                subdomain_id=subdomain_id)
        self.set_kinematics()



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

        self.res_form = 0

        for subdomain in self.subdomains :
            self.res_form += dolfin.inner(
                subdomain.sigma,
                dolfin.sym(dolfin.grad(self.subsols["U"].dsubtest))) * self.dV(subdomain.id)

        for loading in pressure_loadings:
            self.res_form -= dolfin.inner(
               -loading.val * self.mesh_normals,
                self.subsols["U"].dsubtest) * loading.measure

        # dWpordJ = - self.eta / (self.kinematics.Je - self.kinematics.Js)
        # self.res_form += dolfin.inner(
        #     dWpordJ * self.kinematics.Je * self.kinematics.Ce_inv,
        #     dolfin.derivative(
        #             self.kinematics.Et,
        #             self.subsols["U"].subfunc,
        #             self.subsols["U"].dsubtest)) * self.dV

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)


    def add_Phydro_qois(self):

        nb_subdomain = 0
        for subdomain in self.subdomains:
            nb_subdomain += 1
        if nb_subdomain == 1:
            basename = "Phydro_"
            P = -1./3. * dolfin.tr(self.subdomains[0].sigma)

        self.add_qoi(
            name=basename,
            expr=P / self.mesh_V0 * self.dV)



    def add_dPsiBulkdJs_qois(self):

        nb_subdomain = 0
        for subdomain in self.subdomains:
            nb_subdomain += 1
        if nb_subdomain == 1:
            basename = "dPsiBulkdJs_"
            kappa = 10**(9)
            deriv = kappa * (1 / (1 - self.porosity0) - 1 / self.kinematics.Js)

        self.add_qoi(
            name=basename,
            expr=deriv / self.mesh_V0 * self.dV)



    def add_dPsiPordJ_qois(self):

        nb_subdomain = 0
        for subdomain in self.subdomains:
            nb_subdomain += 1
        if nb_subdomain == 1:
            basename = "dPsiPordJ_"
            deriv = - self.eta / (self.kinematics.Je - self.kinematics.Js)

        self.add_qoi(
            name=basename,
            expr=deriv / self.mesh_V0 * self.dV)


    def add_Phi_qois(self):

        basename = "PHI_"
        Phi = self.kinematics.Phi

        self.add_qoi(
            name=basename,
            expr=Phi / self.mesh_V0 * self.dV)



    def add_Js_qois(self):

        basename = "Js_"

        self.add_qoi(
            name=basename,
            expr=self.kinematics.Js / self.mesh_V0 * self.dV)
