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

import dolfin_mech as dmech
from .Material_Elastic_Bulk import BulkElasticMaterial

################################################################################

class SkeletonPoroBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            parameters):

        self.kappa = dolfin.Constant(parameters["kappa"])



    def get_free_energy(self,
            Js,
            Phi0):

        dev_bulk_mat_Js = self.kappa * (1/(1-Phi0) - 1/Js)

        return 0, dev_bulk_mat_Js
