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
from .Material_Elastic import ElasticMaterial

################################################################################

class WporPoroElasticMaterial(ElasticMaterial):



    def __init__(self,
            problem,
            eta,
            type):

        self.problem = problem

        self.eta = eta

        assert type is 'exp' or isinstance(type, int)
        self.type = type



    def get_dWpordJs(self):

        Jf = self.problem.kinematics.Je * self.problem.Phi
        if self.type is 'exp':
            dWpordJs = self.eta * (self.problem.Phi0 / (Jf))**2 * dolfin.exp(-Jf / (self.problem.Phi0 - Jf)) / (self.problem.Phi0 - Jf)
        elif isintance(self.type, int):
            if self.type == 2:
                dWpordJs = self.eta * n * ((self.problem.Phi0 - Jf) / (Jf))**(n-1) * self.problem.Phi0 / (Jf)**2
        dWpordJs = dolfin.conditional(dolfin.lt(self.problem.Phi, self.problem.Phi0), dWpordJs, 0)
        dWpordJs = (1 - self.problem.Phi0) * dWpordJs

        return dWpordJs



###################################################### for mixed formulation ###

    def get_res_term(self):

        dWpordJs = self.get_dWpordJs()

        res_form = dolfin.inner(
                dWpordJs,
                self.problem.subsols["Phi"].dsubtest) * self.problem.dV

        return res_form



########################################## for internal variable formulation ###
