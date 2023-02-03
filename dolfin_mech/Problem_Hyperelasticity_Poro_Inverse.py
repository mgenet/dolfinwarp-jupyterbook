#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Problem_Hyperelasticity_Poro import PoroHyperelasticityProblem
import numpy

################################################################################

class InversePoroHyperelasticityProblem(PoroHyperelasticityProblem):



    def __init__(self, *args, **kwargs):

        PoroHyperelasticityProblem.__init__(self, *args, **kwargs)



    def get_displacement_name(self):

        return "u"



    def get_porosity_name(self):

        return "phis0"



    def set_kinematics(self):

        self.kinematics = dmech.InverseKinematics(
            u=self.get_displacement_subsol().subfunc,
            u_old=self.get_displacement_subsol().func_old)

        self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F")
        self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J")
        self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C")
        self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E")



    def init_known_porosity(self,
            porosity_init_val,
            porosity_init_fun):
        
        # print(porosity_init_fun.vector().get_local())

        if   (porosity_init_val   is not None):
            self.phis = dolfin.Constant(porosity_init_val)
        elif (porosity_init_fun is not None):
            self.phis = porosity_init_fun
            # print(self.phis.vector().get_local())
            # print("phi average", dolfin.assemble(self.phis*self.dV))
        self.add_foi(
            expr=self.phis,
            fs=self.get_porosity_function_space().collapse(),
            name="phis")
        self.add_foi(
            expr=1 - self.phis,
            fs=self.get_porosity_function_space().collapse(),
            name="phif")
        self.add_foi(
            expr=1/self.kinematics.J - self.get_porosity_subsol().subfunc,
            fs=self.get_porosity_function_space().collapse(),
            name="phif0")
        self.add_foi(
            expr=self.kinematics.J * self.get_porosity_subsol().subfunc,
            fs=self.get_porosity_function_space().collapse(),
            name="Phis0")
        self.add_foi(
            expr=1 - self.kinematics.J * self.get_porosity_subsol().subfunc,
            fs=self.get_porosity_function_space().collapse(),
            name="Phif0")

    def get_X0_mass(self):
        self.X0_mass = numpy.empty(self.dim)
        # print("dim=", self.dim)
        for k_dim in range(self.dim):
            self.X0_mass[k_dim] = dolfin.assemble((self.phis*self.X[k_dim])*self.dV)/dolfin.assemble(self.phis*self.dV)
        # print("X0mass", self.X0_mass)
        # self.X0_mass = dolfin.Constant(self.X0_mass)
        return(self.X0_mass)
    
    def get_density(self):
        self.density = self.phis*1e-6
        return(self.density)



    def add_Wskel_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.InverseWskelPoroOperator(
            kinematics=self.kinematics,
            u_test=self.get_displacement_subsol().dsubtest,
            phis0=self.get_porosity_subsol().subfunc,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_Wbulk_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.InverseWbulkPoroOperator(
            kinematics=self.kinematics,
            u_test=self.get_displacement_subsol().dsubtest,
            phis=self.phis,
            phis0=self.get_porosity_subsol().subfunc,
            phis0_test=self.get_porosity_subsol().dsubtest,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)


    def add_Wpore_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.InverseWporePoroOperator(
            kinematics=self.kinematics,
            phis=self.phis,
            phis0=self.get_porosity_subsol().subfunc,
            phis0_test=self.get_porosity_subsol().dsubtest,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_global_porosity_qois(self):

        self.add_qoi(
            name=self.get_porosity_name(),
            expr=self.get_porosity_subsol().subfunc * self.dV)

        self.add_qoi(
            name="phif0",
            expr=(1/self.kinematics.J - self.get_porosity_subsol().subfunc) * self.dV)

        self.add_qoi(
            name="Phis0",
            expr=(self.kinematics.J * self.get_porosity_subsol().subfunc) * self.dV)

        self.add_qoi(
            name="Phif0",
            expr=(1 - self.kinematics.J * self.get_porosity_subsol().subfunc) * self.dV)
