#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_cm as dcm
from Material_Elastic import ElasticMaterial

################################################################################

class PorousMaterial(ElasticMaterial):


    def __init__(self,
                 material,
                 porosity=0):

        self.material = material
        self.porosity = porosity

    def get_free_energy(self,
                        U=None,
                        C=None):

        Psi_mat, Sigma_mat = self.material.get_free_energy(C)
        Psi = (1 - self.porosity) * Psi_mat
        Sigma = (1 - self.porosity) * Sigma_mat

        return Psi, Sigma
