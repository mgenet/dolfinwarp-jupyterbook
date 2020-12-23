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

class PoroConditionProblem(HyperelasticityProblem):



    def __init__(self,
            w_incompressibility=False):

        HyperelasticityProblem.__init__(self,w_incompressibility)



    def set_kinematics(self):

        HyperelasticityProblem.set_kinematics(self)

        # self.kinematics.Phi = 1 - (1 - self.porosity0) / self.kinematics.Je
        # self.kinematics.Js = self.kinematics.Je * (1-self.kinematics.Phi)
        # self.H = dolfin.conditional(dolfin.gt(self.kinematics.Phi * dolfin.conditional(dolfin.gt(self.kinematics.Phi,0),1,0) ,0),1,0)

        self.kinematics.Phi = (1 - (1 - self.porosity0) / self.kinematics.Je) * dolfin.conditional(dolfin.gt(1 - (1 - self.porosity0) / self.kinematics.Je,0),1,0)
        self.kinematics.Js = self.kinematics.Je * (1-self.kinematics.Phi)
        self.H = dolfin.conditional(dolfin.gt(self.kinematics.Phi ,0),1,0)



    def set_materials(self,
            elastic_behavior=None,
            elastic_behavior_dev_skel=None,
            elastic_behavior_bulk_skel=None,
            elastic_behavior_bulk_mat=None,
            subdomain_id=None):

        HyperelasticityProblem.set_materials(self,
                elastic_behavior=elastic_behavior,
                elastic_behavior_dev=elastic_behavior_dev_skel,
                elastic_behavior_bulk=elastic_behavior_bulk_skel,
                subdomain_id=subdomain_id)

        self.Psi_bulk_mat, self.dev_bulk_mat_Js = elastic_behavior_bulk_mat.get_free_energy(
            Js = self.kinematics.Js,
            Phi0 = self.porosity0)

        # self.dev_bulk_mat_Js = self.kappa * ( 1 / (1 - Phi0) - 1 / Js )



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

        self.res_form = dolfin.derivative(
            self.Pi,
            self.sol_func,
            self.dsol_test);

        self.res_form += dolfin.inner(
            (1 - self.H) * self.dev_bulk_mat_Js * self.kinematics.Je * dolfin.inv(self.kinematics.Ce),
            dolfin.derivative(
                    self.kinematics.Et,
                    self.subsols["U"].subfunc,
                    self.subsols["U"].dsubtest)) * self.dV

        for loading in pressure_loadings:
            T = dolfin.dot(
               -loading.val * self.mesh_normals,
                dolfin.inv(self.kinematics.Ft))
            self.res_form -= self.kinematics.Jt * dolfin.inner(
                T,
                self.subsols["U"].dsubtest) * loading.measure

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)



    def add_Phydro_qois(self):

        n_subdomains = len(self.subdomains)
        if (n_subdomains == 1):
            basename = "Phydro_"
            P = -1./3. * dolfin.tr(self.subdomains[0].sigma)

        self.add_qoi(
            name=basename,
            expr=P * self.dV)



    def add_dPsiBulkdJs_qois(self):

        basename = "dPsiBulkdJs_"
        deriv = self.dev_bulk_mat_Js

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



    def add_H_qois(self):

        basename = "H_"

        self.add_qoi(
            name=basename,
            expr=self.H * self.dV)
