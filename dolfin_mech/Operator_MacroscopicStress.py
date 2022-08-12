#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2021-2022                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class MacroscopicStressOperator(Operator):

    def __init__(self,
            mesh_V0,
            mesh_bbox_V0,
            sigma_bar,
            sigma_bar_test,
            material,
            measure):

        self.material = material
        self.measure  = measure

        self.res_form = dolfin.inner((mesh_bbox_V0/mesh_V0) * sigma_bar - self.material.sigma, sigma_bar_test) * self.measure # MG20220426: Need to compute <sigma> properly, including fluid pressure
