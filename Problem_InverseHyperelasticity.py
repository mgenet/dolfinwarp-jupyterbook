#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin
import numpy

import dolfin_cm as dcm
from .Problem import Problem

################################################################################

class InverseHyperelasticityProblem(Problem):



    def __init__(self,
            w_incompressibility=False):

        Problem.__init__(self)

        self.w_incompressibility = w_incompressibility
        assert (not (self.w_incompressibility)), "To do. Aborting."



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
            U_degree=1): #MG20190513: Should have different name, right?

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
            degree=U_degree-1)
        self.set_foi_function_spaces()



    def get_displacement_function_space(self):

        if (len(self.subsols) == 1):
            return self.sol_fs
        else:
            return self.get_subsol_function_space(name="U")



    def set_kinematics(self):

        self.kinematics = dcm.InverseKinematics(
            dim=self.dim,
            U=self.subsols["U"].subfunc,
            U_old=self.subsols["U"].func_old)

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

        self.set_kinematics()

        if (self.w_incompressibility):
            assert (elastic_behavior      is     None)
            assert (elastic_behavior_dev  is not None)
            assert (elastic_behavior_bulk is     None)
        else:
            assert  ((elastic_behavior      is not None)
                or  ((elastic_behavior_dev  is not None)
                and  (elastic_behavior_bulk is not None)))

        subdomain = dcm.SubDomain(
            problem=self,
            elastic_behavior=elastic_behavior,
            elastic_behavior_dev=elastic_behavior_dev,
            elastic_behavior_bulk=elastic_behavior_bulk,
            id=subdomain_id)

        self.subdomains += [subdomain]

        # self.add_foi(expr=subdomain.Sigma, fs=self.mfoi_fs, name="Sigma")
        # self.add_foi(expr=subdomain.PK1  , fs=self.mfoi_fs, name="PK1"  )
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



    def add_J_qois(self,
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
            expr=J / self.mesh_V0 * self.dV)



    def add_P_qois(self):

        nb_subdomain = 0
        for subdomain in self.subdomains:
            nb_subdomain += 1
        # print nb_subdomain

        if nb_subdomain == 0:
             basename = "P_"
             P = -1./3. * dolfin.tr(self.sigma)
        elif nb_subdomain == 1:
            basename = "P_"
            P = -1./3. * dolfin.tr(self.subdomains[0].sigma)

        self.add_qoi(
            name=basename,
            expr=P / self.mesh_V0 * self.dV)
