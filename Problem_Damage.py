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

import dolfin_cm as dcm
from .Problem_Elasticity import ElasticityProblem

################################################################################

class DamageProblem(ElasticityProblem):



    def __init__(self,
            w_damage="mixed"):

        ElasticityProblem.__init__(self,
            w_incompressibility=False)

        self.w_damage = w_damage



    def add_damage_subsol(self,
            degree=0):

        self.add_scalar_subsol(
            name="d",
            family="DG",
            degree=degree)



    def set_subsols(self,
            U_degree=1):

        self.add_displacement_subsol(
            degree=U_degree)

        if (self.w_damage == "mixed"):
            self.add_damage_subsol()



    def set_materials(self,
            damaged_behavior,
            damage_behavior):

        if   (self.w_damage == "mixed"):
            self.psi, self.sigma = damaged_behavior.get_free_energy(
                epsilon=self.kinematics.epsilon,
                d=self.subsols["d"].subfunc)
            self.inelastic_behaviors_mixed += [damage_behavior]
        elif (self.w_damage == "internal"):
            damage_behavior.set_internal_variables_internal()
            self.psi, self.sigma = damaged_behavior.get_free_energy(
                epsilon=self.kinematics.epsilon,
                d=damage_behavior.d)
            self.inelastic_behaviors_internal += [damage_behavior]

        self.add_foi(expr=self.sigma, fs=self.mfoi_fs, name="sigma")



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

        self.res_form = dolfin.inner(
            self.sigma,
            dolfin.derivative(
                self.kinematics.epsilon,
                self.subsols["u"].subfunc,
                self.subsols["u"].dsubtest)) * self.dV

        for loading in surface_loadings:
            self.res_form -= dolfin.inner(
                loading.val,
                self.subsols["u"].dsubtest) * loading.measure

        for loading in pressure_loadings:
            self.res_form -= dolfin.inner(
               -loading.val * self.mesh_normals,
                self.subsols["u"].dsubtest) * loading.measure

        for loading in volume_loadings:
            self.res_form -= dolfin.inner(
                loading.val,
                self.subsols["u"].dsubtest) * loading.measure

        for inelastic_behavior in self.inelastic_behaviors_mixed:
            self.res_form += inelastic_behavior.get_res_term(
                dt=dt)

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)



    def add_global_stress_qois(self):

        basename = "s_"
        stress = self.sigma

        self.add_qoi(
            name=basename+"XX",
            expr=stress[0,0] * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"YY",
                expr=stress[1,1] * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"ZZ",
                    expr=stress[2,2] * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"XY",
                expr=stress[0,1] * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"YZ",
                    expr=stress[1,2] * self.dV)
                self.add_qoi(
                    name=basename+"ZX",
                    expr=stress[2,0] * self.dV)
