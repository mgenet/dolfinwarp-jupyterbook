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

import dolfin_mech as dmech
from .Material_Elastic import ElasticMaterial

################################################################################

class WporPoroElasticMaterial(ElasticMaterial):



    def __init__(self,
            problem,
            parameters,
            type):

        self.problem = problem

        assert ('eta' in parameters)
        self.eta = parameters['eta']

        assert ((type == 'exp') or isinstance(type, int))
        self.type = type



    def get_dWpordJs(self):

        Jf = self.problem.kinematics.Je * self.problem.Phi
        if (self.type == 'exp'):
            dWpordJs = self.eta * (self.problem.Phi0 / (Jf))**2 * dolfin.exp(-Jf / (self.problem.Phi0 - Jf)) / (self.problem.Phi0 - Jf)
        elif (isintance(self.type, int)):
            if (self.type == 2):
                dWpordJs = self.eta * n * ((self.problem.Phi0 - Jf) / (Jf))**(n-1) * self.problem.Phi0 / (Jf)**2
        dWpordJs = dolfin.conditional(dolfin.lt(self.problem.Phi, self.problem.Phi0), dWpordJs, 0)
        dWpordJs = (1 - self.problem.Phi0) * dWpordJs

        return dWpordJs



###################################################### for mixed formulation ###

    def get_res_term(self,
			w_Phi0=None,
			w_Phi=None):

        assert ((w_Phi0 is     None) or (w_Phi is     None))
        assert ((w_Phi0 is not None) or (w_Phi is not None))

        dWpordJs = self.get_dWpordJs()

        if (w_Phi0 is not None):
            res_form = dolfin.inner(
				dWpordJs,
				self.problem.subsols["Phi0"].dsubtest) * self.problem.dV
        elif (w_Phi is not None):
            res_form = dolfin.inner(
				dWpordJs,
				self.problem.subsols["Phi"].dsubtest) * self.problem.dV

        return res_form



########################################## for internal variable formulation ###
