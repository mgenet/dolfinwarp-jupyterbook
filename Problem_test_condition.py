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
from .Problem_Hyperelasticity import HyperelasticityProblem

################################################################################

class TestConditionProblem(HyperelasticityProblem):



    def __init__(self,
            w_incompressibility=False):

        HyperelasticityProblem.__init__(self,w_incompressibility)



    def set_materials(self,
            elastic_behavior_dev1=None,
            elastic_behavior_bulk1=None,
            elastic_behavior_dev2=None,
            elastic_behavior_bulk2=None,
            subdomain_id=None):

        self.Psi_bulk1, self.Sigma_bulk1 = elastic_behavior_bulk1.get_free_energy(
            C=self.kinematics.Ce)
        self.Psi_dev1, self.Sigma_dev1 = elastic_behavior_dev1.get_free_energy(
            C=self.kinematics.Ce)
        self.Psi1   = self.Psi_bulk1   + self.Psi_dev1
        self.Sigma1 = self.Sigma_bulk1 + self.Sigma_dev1

        self.Psi_bulk2, self.Sigma_bulk2 = elastic_behavior_bulk2.get_free_energy(
            C=self.kinematics.Ce)
        self.Psi_dev2, self.Sigma_dev2 = elastic_behavior_dev2.get_free_energy(
            C=self.kinematics.Ce)
        self.Psi2   = self.Psi_bulk2   + self.Psi_dev2
        self.Sigma2 = self.Sigma_bulk2 + self.Sigma_dev2

        self.Sigma = self.Sigma1
        self.PK1   = self.kinematics.Ft * self.Sigma
        self.sigma = (1./self.kinematics.Jt) * self.PK1 * self.kinematics.Ft.T

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

        # self.Pi = sum([subdomain.Psi * self.dV(subdomain.id) for subdomain in self.subdomains])

        self.H = dolfin.conditional(dolfin.gt(self.kinematics.Je,0.8),1,0)
        self.Pi = self.H * self.Psi1 * self.dV + (1 - self.H) * self.Psi2 * self.dV

        self.Sigma = self.H * self.Sigma1 + (1 - self.H) * self.Sigma2
        self.PK1   = self.kinematics.Ft * self.Sigma
        self.sigma = (1./self.kinematics.Jt) * self.PK1 * self.kinematics.Ft.T

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

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)
