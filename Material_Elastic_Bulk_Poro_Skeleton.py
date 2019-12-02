#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019                                                   ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin

import dolfin_cm as dcm
from .Material_Elastic_Bulk import BulkElasticMaterial

################################################################################

class SkeletonPoroBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            parameters):

        self.kappa = dolfin.Constant(parameters["kappa"])



    def get_free_energy(self,
            Js=None,
            Phi0=None,
            Phi=None):

        # Psi   =
        dev_bulk_mat_Js = self.kappa * ( 1 / (1 - Phi0) - 1 / Js )

        return 0, dev_bulk_mat_Js
        # return Psi, Sigma
