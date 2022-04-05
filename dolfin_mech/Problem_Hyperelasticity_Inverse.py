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
from .Problem_Hyperelasticity import HyperelasticityProblem

################################################################################

class InverseHyperelasticityProblem(HyperelasticityProblem):



    def __init__(self, *args, **kwargs):

        if ("w_incompressibility" in kwargs):
            assert (bool(kwargs.w_incompressibility) == 0),\
                "Incompressibility not implemented for inverse problem. Aborting."

        HyperelasticityProblem.__init__(self, *args, **kwargs)



    def set_kinematics(self):

        self.kinematics = dmech.InverseKinematics(
            U=self.get_displacement_subsol().subfunc,
            U_old=self.get_displacement_subsol().func_old)

        self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F")
        self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J")
        self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C")
        self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E")



    def add_elasticity_operator(self,
            material_model,
            material_parameters,
            subdomain_id=None):

        operator = dmech.LinearizedElasticityOperator(
            kinematics=self.kinematics,
            u_test=self.get_displacement_subsol().dsubtest,
            material_model=material_model,
            material_parameters=material_parameters,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)
