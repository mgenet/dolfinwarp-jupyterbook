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
            eta,
            kappa):

        HyperelasticityProblem.__init__(self,w_incompressibility=False)
        self.eta = eta
        self.kappa = kappa



    def set_porosity_energy(self):

        # Wpor = - self.eta * dolfin.log(self.kinematics.Je - self.kinematics.Js)
        dWpordJ = - self.eta / (self.kinematics.Je - self.kinematics.Js)
        # self.dWpordJ = dWpordJ
        self.dWpordJ = self.coef_1_minus_phi0 * dWpordJ

        if self.config_porosity == 'ref':
            if self.kappa == 0:
                Js0 = self.coef_1_minus_phi0
            else:
                Js0 = self.coef_1_minus_phi0/2 * (1 + 1/self.coef_1_minus_phi0 + self.eta/self.kappa - ((1 + 1/self.coef_1_minus_phi0 + self.eta/self.kappa)**2 - 4 / self.coef_1_minus_phi0)**(1./2.))
            self.dWpordJ += self.coef_1_minus_phi0 * self.eta / (1 - Js0)



    def set_coef_1_minus_phi0(self,
            config_porosity='ref'):

        if self.config_porosity == 'ref':
            coef = 1 - self.porosity_given
        elif self.config_porosity == 'deformed':
            coef = Nan

        self.coef_1_minus_phi0 = coef



    def set_kinematics(self):

        HyperelasticityProblem.set_kinematics(self)

        self.set_coef_1_minus_phi0(self.config_porosity)

        if self.config_porosity == 'ref':
            # first solution
            # self.kinematics.Phi = 1 - (1 - self.porosity0) / self.kinematics.Je
            # self.kinematics.Js = self.kinematics.Je * (1-self.kinematics.Phi)

            # the same
            # self.kinematics.Phi = 1 - (1 - self.porosity0) / self.kinematics.Je
            # self.kinematics.Js = 1 - self.porosity0

            # better
            if self.eta == 0:
                self.kinematics.Js = self.coef_1_minus_phi0
                self.kinematics.Phi = 1 - self.kinematics.Js / self.kinematics.Je

            # first way to write
            # delta = (self.eta + self.kappa*(1 + (self.kinematics.Je / (1-self.porosity0))**(1./2.))**2) * (self.eta + self.kappa*(1 - (self.kinematics.Je / (1-self.porosity0))**(1./2.))**2)
            # self.kinematics.Js = (1-self.porosity0)/(2*self.kappa) * (self.eta + self.kappa*(1+self.kinematics.Je/(1-self.porosity0)) - delta**(1./2.))
            # self.kinematics.Phi = 1 - self.kinematics.Js / self.kinematics.Je

            # second way to write (kappa)
            else:
                self.kinematics.Js = self.coef_1_minus_phi0/2 * (1 + self.kinematics.Je/self.coef_1_minus_phi0 + self.eta/self.kappa - ((1 + self.kinematics.Je/self.coef_1_minus_phi0 + self.eta/self.kappa)**2 - 4*self.kinematics.Je / self.coef_1_minus_phi0)**(1./2.))
                self.kinematics.Phi = 1 - self.kinematics.Js / self.kinematics.Je
        else:
            self.kinematics.Js = Nan
            self.kinematics.Phi = Nan



    def set_materials(self,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None,
            subdomain_id=None):

        self.set_kinematics()

        HyperelasticityProblem.set_materials(self,
                elastic_behavior=elastic_behavior,
                elastic_behavior_dev=elastic_behavior_dev,
                elastic_behavior_bulk=elastic_behavior_bulk,
                subdomain_id=subdomain_id)

        self.set_porosity_energy()



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

        self.res_form += dolfin.inner(
            self.dWpordJ * self.kinematics.Je * self.kinematics.Ce_inv,
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
            expr=P / self.mesh_V0 * self.dV)



    # def add_dPsiBulkdJs_qois(self):
    #
    #     nb_subdomain = 0
    #     for subdomain in self.subdomains:
    #         nb_subdomain += 1
    #     if nb_subdomain == 1:
    #         basename = "dPsiBulkdJs_"
    #         deriv = self.kappa * (1 / (1 - self.porosity0) - 1 / self.kinematics.Js)
    #
    #     self.add_qoi(
    #         name=basename,
    #         expr=deriv / self.mesh_V0 * self.dV)



    # def add_dPsiPordJ_qois(self):
    #
    #     nb_subdomain = 0
    #     for subdomain in self.subdomains:
    #         nb_subdomain += 1
    #     if nb_subdomain == 1:
    #         basename = "dPsiPordJ_"
    #         deriv = - self.eta / (self.kinematics.Je - self.kinematics.Js)
    #
    #     self.add_qoi(
    #         name=basename,
    #         expr=deriv / self.mesh_V0 * self.dV)


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



    def add_coef_qois(self):

        basename = "coef_"

        self.add_qoi(
            name=basename,
            expr=self.coef_1_minus_phi0 / self.mesh_V0 * self.dV)
