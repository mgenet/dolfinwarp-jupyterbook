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

class HyperelasticityProblem(Problem):



    def __init__(self,
            w_incompressibility=False):

        Problem.__init__(self)

        self.w_incompressibility = w_incompressibility



    def add_displacement_subsol(self,
            degree):

        self.add_vector_subsol(
            name="U",
            family="CG",
            degree=degree)



    def add_pressure_subsol(self,
            degree):

        if (degree == 0):
            self.add_scalar_subsol(
                name="P",
                family="DG",
                degree=0)
        else:
            self.add_scalar_subsol(
                name="P",
                family="CG",
                degree=degree)



    def set_subsols(self,
            U_degree=1):

        self.add_displacement_subsol(
            degree=U_degree)

        if (self.w_incompressibility):
            self.add_pressure_subsol(
                degree=U_degree-1)



    def set_solution_degree(self,
            U_degree=1): # MG20190513: Should have different name, right?

        self.set_subsols(
            U_degree=U_degree)
        self.set_solution_finite_element()
        self.set_solution_function_space()
        self.set_solution_functions()

        if (self.mesh.ufl_cell().cellname() in ("triangle", "tetrahedron")):
            quadrature_degree = max(1, 2*(U_degree-1))
        elif (self.mesh.ufl_cell().cellname() in ("quadrilateral", "hexahedron")):
            quadrature_degree = max(1, 2*(self.dim*U_degree-1))
        self.set_quadrature_degree(
            quadrature_degree=quadrature_degree)

        self.set_foi_finite_elements_DG(
            degree=0)
        self.set_foi_function_spaces()



    def get_displacement_function_space(self):

        if (len(self.subsols) == 1):
            return self.sol_fs
        else:
            return self.get_subsol_function_space(name="U")



    def get_pressure_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="P")



    def set_kinematics(self):

        self.kinematics = dmech.Kinematics(
            dim=self.dim,
            U=self.subsols["U"].subfunc,
            U_old=self.subsols["U"].func_old,
            Q_expr=self.Q_expr)

        self.add_foi(expr=self.kinematics.Fe, fs=self.mfoi_fs, name="F")
        self.add_foi(expr=self.kinematics.Je, fs=self.sfoi_fs, name="J")
        self.add_foi(expr=self.kinematics.Ce, fs=self.mfoi_fs, name="C")
        self.add_foi(expr=self.kinematics.Ee, fs=self.mfoi_fs, name="E")
        if (self.Q_expr is not None):
            self.add_foi(expr=self.kinematics.Ee_loc, fs=self.mfoi_fs, name="Ee_loc")



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

        subdomain = dmech.SubDomain(
            problem=self,
            elastic_behavior=elastic_behavior,
            elastic_behavior_dev=elastic_behavior_dev,
            elastic_behavior_bulk=elastic_behavior_bulk,
            id=subdomain_id)
        self.subdomains += [subdomain]

        self.add_foi(expr=subdomain.Sigma, fs=self.mfoi_fs, name="Sigma")
        # self.add_foi(expr=subdomain.PK1  , fs=self.mfoi_fs, name="PK1"  )
        self.add_foi(expr=subdomain.sigma, fs=self.mfoi_fs, name="sigma")

        if (self.Q_expr is not None):
            subdomain.sigma_loc = dolfin.dot(dolfin.dot(self.Q_expr, subdomain.sigma), self.Q_expr.T)
            self.add_foi(expr=subdomain.sigma_loc, fs=self.mfoi_fs, name="sigma_loc")



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

        self.Pi = sum([subdomain.Psi * self.dV(subdomain.id) for subdomain in self.subdomains])
        # print (self.Pi)

        for loading in normal_penalties:
            self.Pi += (loading.val/2) * dolfin.inner(
                self.subsols["U"].subfunc,
                self.mesh_normals)**2 * loading.measure

        # for loading in directional_penalties: # MG20190513: Cannot use point integral within assemble_system
        #     self.Pi += (loading.val/2) * dolfin.inner(
        #         self.subsols["U"].subfunc,
        #         loading.N)**2 * loading.measure

        for loading in surface_tensions:
            FmTN = dolfin.dot(
                dolfin.inv(self.kinematics.Ft).T,
                self.mesh_normals)
            T = dolfin.sqrt(dolfin.inner(
                FmTN,
                FmTN))
            self.Pi += loading.val * self.kinematics.Jt * T * loading.measure

        for loading in surface0_loadings:
            self.Pi -= dolfin.inner(
                loading.val,
                self.subsols["U"].subfunc) * loading.measure

        for loading in pressure0_loadings:
            self.Pi -= dolfin.inner(
               -loading.val * self.mesh_normals,
                self.subsols["U"].subfunc) * loading.measure

        for loading in volume0_loadings:
            self.Pi -= dolfin.inner(
                loading.val,
                self.subsols["U"].subfunc) * loading.measure

        self.res_form = dolfin.derivative(
            self.Pi,
            self.sol_func,
            self.dsol_test)

        # self.res_form += dolfin.inner(
        #     self.Sigma,
        #     dolfin.derivative(
        #         self.kinematics.Et,
        #         self.subsols["U"].subfunc,
        #         self.subsols["U"].dsubtest)) * self.dV
        #
        # if (self.w_incompressibility):
        #     self.res_form += dolfin.inner(
        #         self.kinematics.Je-1,
        #         self.subsols["P"].dsubtest) * self.dV

        # for loading in normal_penalties:
        #     self.res_form += loading.val * dolfin.inner(
        #         self.subsols["U"].subfunc,
        #         self.mesh_normals) * dolfin.inner(
        #         self.subsols["U"].dsubtest,
        #         self.mesh_normals) * loading.measure

        # for loading in surface0_loadings:
        #     self.res_form -= dolfin.inner(
        #         loading.val,
        #         self.subsols["U"].dsubtest) * loading.measure
        #
        # for loading in pressure0_loadings:
        #     self.res_form -= dolfin.inner(
        #        -loading.val * self.mesh_normals,
        #         self.subsols["U"].dsubtest) * loading.measure
        #
        # for loading in volume0_loadings:
        #     self.res_form -= dolfin.inner(
        #         loading.val,
        #         self.subsols["U"].dsubtest) * loading.measure

        for loading in surface_loadings:
            FmTN = dolfin.dot(
                dolfin.inv(self.kinematics.Ft).T,
                self.mesh_normals)
            T = dolfin.sqrt(dolfin.inner(
                FmTN,
                FmTN)) * loading.val
            self.res_form -= self.kinematics.Jt * dolfin.inner(
                T,
                self.subsols["U"].dsubtest) * loading.measure

        for loading in pressure_loadings:
            T = dolfin.dot(
               -loading.val * self.mesh_normals,
                dolfin.inv(self.kinematics.Ft))
            self.res_form -= self.kinematics.Jt * dolfin.inner(
                T,
                self.subsols["U"].dsubtest) * loading.measure

        for loading in volume_loadings:
            self.res_form -= self.kinematics.Jt * dolfin.inner(
                loading.val,
                self.subsols["U"].dsubtest) * loading.measure

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)



    def add_global_strain_qois(self,
            strain_type="elastic",
            configuration_type="loaded"):

        if (configuration_type == "loaded"):
            kin = self.kinematics
        elif (configuration_type == "unloaded"):
            kin = self.unloaded_kinematics

        if (strain_type == "elastic"):
            basename = "E^e_"
            strain = kin.Ee
        elif (strain_type == "total"):
            basename = "E^t_"
            strain = kin.Et

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



    def add_global_volume_ratio_qois(self,
            J_type="elastic",
            configuration_type="loaded"):

        if (configuration_type == "loaded"):
            kin = self.kinematics
        elif (configuration_type == "unloaded"):
            kin = self.unloaded_kinematics

        if (J_type == "elastic"):
            basename = "J^e_"
            J = kin.Je
        elif (J_type == "total"):
            basename = "J^t_"
            J = kin.Jt

        self.add_qoi(
            name=basename,
            expr=J * self.dV)



    def add_global_stress_qois(self,
            stress_type="cauchy"):

        n_subdomains = len(self.subdomains)
        for subdomain in self.subdomains:
            if (stress_type in ("Cauchy", "cauchy", "sigma")):
                basename = "s_"
                stress = subdomain.sigma
            elif (stress_type in ("Piola", "piola", "PK2", "Sigma")):
                basename = "S_"
                stress = subdomain.Sigma
            elif (stress_type in ("Boussinesq", "boussinesq", "PK1", "P")):
                basename = "P_"
                stress = subdomain.PK1

            if (n_subdomains > 1):
                basename += str(subdomain.id)+"_"

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
