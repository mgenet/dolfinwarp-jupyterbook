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

class ElasticityProblem(Problem):



    def __init__(self,
            w_incompressibility=False):

        Problem.__init__(self)

        self.w_incompressibility = w_incompressibility



    def add_displacement_subsol(self,
            degree):

        self.add_vector_subsol(
            name="u",
            family="CG",
            degree=degree)



    def add_pressure_subsol(self,
            degree):

        if (degree == 0):
            self.add_scalar_subsol(
                name="p",
                family="DG",
                degree=0)
        else:
            self.add_scalar_subsol(
                name="p",
                family="CG",
                degree=degree)



    def set_subsols(self,
            U_degree=1):

        self.add_displacement_subsol(
            degree=U_degree)

        if (self.w_incompressibility):
            self.add_pressure_subsol(
                degree=U_degree-1)



    def get_displacement_function_space(self):

        if (len(self.subsols) == 1):
            return self.sol_fs
        else:
            return self.get_subsol_function_space(name="u")



    def get_pressure_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="p")



    def set_kinematics(self):

        self.kinematics = dmech.LinearizedKinematics(
            dim=self.dim,
            U=self.subsols["u"].subfunc,
            U_old=self.subsols["u"].func_old)

        self.add_foi(expr=self.kinematics.epsilon, fs=self.mfoi_fs, name="epsilon")



    def set_materials(self,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None,
            subdomain_id=None):

        if (self.w_incompressibility):
            assert (elastic_behavior      is     None)
            assert (elastic_behavior_dev  is not None)
            assert (elastic_behavior_bulk is     None)
        else:
            assert  ((elastic_behavior      is not None)
                or  ((elastic_behavior_dev  is not None)
                and  (elastic_behavior_bulk is not None)))

        subdomain = dmech.LinearizedSubDomain(
            problem=self,
            elastic_behavior=elastic_behavior,
            elastic_behavior_dev=elastic_behavior_dev,
            elastic_behavior_bulk=elastic_behavior_bulk,
            id=subdomain_id)
        self.subdomains += [subdomain]

        self.add_foi(expr=subdomain.sigma, fs=self.mfoi_fs, name="sigma")



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

        self.Pi = sum([subdomain.psi * self.dV(subdomain.id) for subdomain in self.subdomains])
        # print (self.Pi)

        for loading in surface_loadings:
            self.Pi -= dolfin.inner(
                loading.val,
                self.subsols["u"].subfunc) * loading.measure

        for loading in pressure_loadings:
            self.Pi -= dolfin.inner(
               -loading.val * self.mesh_normals,
                self.subsols["u"].subfunc) * loading.measure

        for loading in volume_loadings:
            self.Pi -= dolfin.inner(
                loading.val,
                self.subsols["u"].subfunc) * loading.measure

        self.res_form = dolfin.derivative(
            self.Pi,
            self.sol_func,
            self.dsol_test)

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)



    def add_global_strain_qois(self):

        basename = "e_"
        strain = self.kinematics.epsilon

        self.add_qoi(
            name=basename+"XX",
            expr=strain[0,0] * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"YY",
                expr=strain[1,1] * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"ZZ",
                    expr=strain[2,2] * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"XY",
                expr=strain[0,1] * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"YZ",
                    expr=strain[1,2] * self.dV)
                self.add_qoi(
                    name=basename+"ZX",
                    expr=strain[2,0] * self.dV)



    def add_global_stress_qois(self):

        n_subdomains = len(self.subdomains)
        for subdomain in self.subdomains:
            basename = "s_"
            if (n_subdomains > 1):
                basename += str(subdomain.id)+"_"

            stress = subdomain.sigma

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



    def add_global_pressure_qois(self):

        n_subdomains = len(self.subdomains)
        for subdomain in self.subdomains:

            basename = "p_"
            if (n_subdomains > 1):
                basename += str(subdomain.id)+"_"

            p = -1./3. * dolfin.tr(subdomain.sigma)

            self.add_qoi(
                name=basename,
                expr=p * self.dV)
