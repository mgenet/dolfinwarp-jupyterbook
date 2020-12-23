#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2020                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin
import numpy

import dolfin_mech as dmech
from .Problem import Problem

################################################################################

class InverseKinematics():



    def __init__(self,
            dim,
            U,
            U_old):

        self.I = dolfin.Identity(dim)

        self.ft     = self.I + dolfin.grad(U    )
        self.ft_old = self.I + dolfin.grad(U_old)

        self.Ft     = dolfin.inv(self.ft    )
        self.Ft_old = dolfin.inv(self.ft_old)

        self.Jt     = dolfin.det(self.Ft    )
        self.Jt_old = dolfin.det(self.Ft_old)

        self.Ct     = self.Ft.T     * self.Ft
        self.Ct_old = self.Ft_old.T * self.Ft_old

        self.Et     = (self.Ct     - self.I)/2
        self.Et_old = (self.Ct_old - self.I)/2

        self.Fe     = self.Ft
        self.Fe_old = self.Ft_old

        self.Je     = dolfin.det(self.Fe    )
        self.Je_old = dolfin.det(self.Fe_old)

        self.Ce     = self.Fe.T     * self.Fe
        self.Ce_old = self.Fe_old.T * self.Fe_old
        self.Ce_inv = dolfin.inv(self.Ce)
        self.ICe    = dolfin.tr(self.Ce)

        self.Ee     = (self.Ce     - self.I)/2
        self.Ee_old = (self.Ce_old - self.I)/2

        self.Fe_bar  = self.Je**(-1./3) * self.Fe
        self.Ce_bar  = self.Fe_bar.T * self.Fe_bar
        self.ICe_bar = dolfin.tr(self.Ce_bar)
