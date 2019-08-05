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
                        *args,
                        **kwargs):

        Psi_mat, Sigma_mat = self.material.get_free_energy(*args,
                                                           **kwargs)
        Psi = [(1 - self.porosity) * Psi_k for Psi_k in Psi_mat]
        Sigma = [(1 - self.porosity) * Sigma_k for Sigma_k in Sigma_mat]

        return Psi, Sigma
