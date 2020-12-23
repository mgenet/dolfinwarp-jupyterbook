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
from .Problem_Hyperelasticity import HyperelasticityProblem

################################################################################

class InverseHyperelasticityProblem(HyperelasticityProblem):



    def __init__(self,
            w_incompressibility=False):

        HyperelasticityProblem.__init__(self)

        self.w_incompressibility = w_incompressibility
        assert (not (self.w_incompressibility)), "To do. Aborting."



    def set_kinematics(self):

        self.kinematics = dmech.InverseKinematics(
            dim=self.dim,
            U=self.subsols["U"].subfunc,
            U_old=self.subsols["U"].func_old)

        self.add_foi(expr=self.kinematics.Fe, fs=self.mfoi_fs, name="F")
        self.add_foi(expr=self.kinematics.Je, fs=self.sfoi_fs, name="J")
        self.add_foi(expr=self.kinematics.Ce, fs=self.mfoi_fs, name="C")
        self.add_foi(expr=self.kinematics.Ee, fs=self.mfoi_fs, name="E")
        if (self.Q_expr is not None):
            self.add_foi(expr=self.kinematics.Ee_loc, fs=self.mfoi_fs, name="Ee_loc")



    def set_variational_formulation(self,
            normal_penalties=[],
            directional_penalties=[],
            surface_tensions=[],
            surface0_loadings=[],
            pressure0_loadings=[],
            volume0_loadings=[],
            surface_loadings=[],
            pressure_loadings=[],
            volume_loadings=[],
            dt=None):

        # self.res_form = 0.                            # MG20190417: ok??
        # self.res_form = dolfin.Constant(0.) * self.dV # MG20190417: arity mismatch??

        self.res_form = 0
        for subdomain in self.subdomains :
            self.res_form += dolfin.inner(
                subdomain.sigma,
                dolfin.sym(dolfin.grad(self.subsols["U"].dsubtest))) * self.dV(subdomain.id)

        if (self.w_incompressibility):
            self.res_form += dolfin.inner(
                self.kinematics.Je-1,
                self.subsols["P"].dsubtest) * self.dV

        for loading in surface_loadings:
            self.res_form -= dolfin.inner(
                loading.val,
                self.subsols["U"].dsubtest) * loading.measure

        for loading in pressure_loadings:
            self.res_form -= dolfin.inner(
               -loading.val * self.mesh_normals,
                self.subsols["U"].dsubtest) * loading.measure

        for loading in volume_loadings:
            self.res_form -= dolfin.inner(
                loading.val,
                self.subsols["U"].dsubtest) * loading.measure

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)
