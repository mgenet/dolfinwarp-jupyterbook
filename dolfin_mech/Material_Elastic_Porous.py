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

import dolfin_mech as dmech
from .Material_Elastic import ElasticMaterial

################################################################################

class PorousMaterial(ElasticMaterial):



    def __init__(self,
            material,
            porosity=0,
            config_porosity='ref'):

        self.material        = material
        self.porosity        = porosity
        self.config_porosity = config_porosity



    def get_free_energy(self,
            C):

        Psi_mat, Sigma_mat = self.material.get_free_energy(
            C=C)
        if   (self.config_porosity == 'ref'):
            Psi   = (1 - self.porosity) * Psi_mat
            Sigma = (1 - self.porosity) * Sigma_mat
        elif (self.config_porosity == 'deformed'):
            J = dolfin.sqrt(dolfin.det(C))
            Psi   = (1 - self.porosity) * J * Psi_mat
            Sigma = (1 - self.porosity) * J * Sigma_mat

        return Psi, Sigma
