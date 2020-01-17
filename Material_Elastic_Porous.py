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

import dolfin

import dolfin_cm as dcm
from .Material_Elastic import ElasticMaterial

################################################################################

class PorousMaterial(ElasticMaterial):



    def __init__(self,
            material,
            problem,
            porosity=0,
            config_porosity='ref'):

        self.material        = material
        self.problem         = problem
        self.porosity_given  = porosity
        self.config_porosity = config_porosity



    def get_free_energy(self,
            C):

        Psi_mat, Sigma_mat = self.material.get_free_energy(C=C)

        if isinstance(self.problem, dcm.TwoSubfuncPoroProblem) or isinstance(self.problem, dcm.TwoSubfuncInversePoroProblem) or isinstance(self.problem, dcm.CondTwoSubfuncPoroProblem):
            assert 'Phi0pos' in self.problem.__dict__
            assert 'coef_1_minus_phi0' not in self.problem.__dict__

        if 'coef_1_minus_phi0' in self.problem.__dict__:
            Psi   = self.problem.coef_1_minus_phi0 * Psi_mat
            Sigma = self.problem.coef_1_minus_phi0 * Sigma_mat
        elif 'Phi0pos' in self.problem.__dict__:
            Psi   = (1 - self.problem.Phi0pos) * Psi_mat
            Sigma = (1 - self.problem.Phi0pos) * Sigma_mat
        else:
            if self.config_porosity == 'ref':
                Psi   = (1-self.porosity_given) * Psi_mat
                Sigma = (1-self.porosity_given) * Sigma_mat
            elif self.config_porosity == 'deformed':
                assert C is not None
                J = dolfin.sqrt(dolfin.det(C))
                Psi   = (1-self.porosity_given) * J * Psi_mat
                Sigma = (1-self.porosity_given) * J * Sigma_mat

        return Psi, Sigma
