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



    def __init__(self,
            *args,
            **kwargs):

        if ("w_incompressibility" in kwargs):
            assert (bool(kwargs.w_incompressibility) == 0),\
                "Incompressibility not implemented for inverse problem. Aborting."

        HyperelasticityProblem.__init__(self, *args, **kwargs)



    def set_kinematics(self):

        self.kinematics = dmech.InverseKinematics(
            dim=self.dim,
            U=self.subsols["U"].subfunc,
            U_old=self.subsols["U"].func_old)

        self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F")
        self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J")
        self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C")
        self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E")



    def add_elasticity_operator(self,
            elastic_behavior,
            subdomain_id=None):

        if (subdomain_id is None):
            measure = self.dV
        else:
            measure = self.dV(subdomain_id)
        operator = dmech.InverseElasticityOperator(
            U_test=self.get_displacement_subsol().dsubtest,
            kinematics=self.kinematics,
            elastic_behavior=elastic_behavior,
            measure=measure)
        return self.add_operator(operator)



    # def add_global_volume_ratio_qois(self,
    #         J_type="elastic",
    #         configuration_type="loaded",
    #         id_zone=None):
    #     if (configuration_type == "loaded"):
    #         kin = self.kinematics
    #     elif (configuration_type == "unloaded"):
    #         kin = self.unloaded_kinematics
    #     if (J_type == "elastic"):
    #         basename = "J^e_"
    #         J = kin.Je
    #     elif (J_type == "total"):
    #         basename = "J^t_"
    #         J = kin.Jt
    #     if id_zone == None:
    #         self.add_qoi(
    #             name=basename,
    #             expr=J / self.mesh_V0 * self.dV)
    #     else:
    #         self.add_qoi(
    #             name=basename,
    #             expr=J / self.mesh_V0 * self.dV(id_zone))



    # def add_Phi0_qois(self):
    #     basename = "PHI0_"
    #     PHI0 = 1 - self.kinematics.Je * (1 - self.porosity_given)
    #     self.add_qoi(
    #         name=basename,
    #         expr=PHI0 / self.mesh_V0 * self.dV)



    # def add_Phi_qois(self):
    #     basename = "PHI_"
    #     PHI = self.porosity_given
    #     self.add_qoi(
    #         name=basename,
    #         expr=PHI / self.mesh_V0 * self.dV)
