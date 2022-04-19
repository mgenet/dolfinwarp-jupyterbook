#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech

################################################################################

class DiluteScheme():



    def __init__(self,
            k_s,
            mu_s,
            phi):

        self.k_s = k_s
        self.mu_s = mu_s
        self.phi = phi        

    def Dulite_equation(self, mu_hom):
        return ((1+4*self.mu_s/(3*self.k_s))*(mu_hom/self.mu_s)**3)/(2 + (4*self.mu_s/(3*self.k_s) - 1)*(mu_hom/self.mu_s)**(3/5)) - (1 - self.phi)**6


    def Dormieux_parameter(self):
        h = 10e-6
        mu_hom = self.mu_s  #Initialization
        res = 1
        while (res > 10e-7):
            res = self.Dulite_equation(mu_hom)
            jac = (self.Dulite_equation(mu_hom + h) - self.Dulite_equation(mu_hom,))/h
            delta = - res/jac
            mu_hom = mu_hom + delta

        return mu_hom
