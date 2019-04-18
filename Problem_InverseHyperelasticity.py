#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_cm as dcm
from Problem import Problem

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
            U_degree=1):

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



    def set_materials(self,
            elastic_behavior_dev,
            elastic_behavior_bulk=None):

        self.elastic_behavior_dev = elastic_behavior_dev

        if not (self.w_incompressibility):
            assert (elastic_behavior_bulk is not None)
            self.elastic_behavior_bulk = elastic_behavior_bulk

        self.kinematics = dcm.InverseKinematics(
            dim=self.dim,
            U=self.subsols["U"].subfunc,
            U_old=self.subsols["U"].func_old)

        self.add_foi(expr=self.kinematics.Fe, fs=self.mfoi_fs, name="F")
        self.add_foi(expr=self.kinematics.Je, fs=self.sfoi_fs, name="J")
        self.add_foi(expr=self.kinematics.Ce, fs=self.mfoi_fs, name="C")
        self.add_foi(expr=self.kinematics.Ee, fs=self.mfoi_fs, name="E")

        if (self.w_incompressibility):
            self.Psi_bulk   = -self.subsols["P"].subfunc * (self.kinematics.Je - 1)
            self.Sigma_bulk = -self.subsols["P"].subfunc *  self.kinematics.Je      * self.kinematics.Ce_inv
        else:
            self.Psi_bulk,\
            self.Sigma_bulk = self.elastic_behavior_bulk.get_free_energy(
                C=self.kinematics.Ce)
        self.Psi_dev,\
        self.Sigma_dev = self.elastic_behavior_dev.get_free_energy(
            C=self.kinematics.Ce)
        self.Sigma = self.Sigma_bulk + self.Sigma_dev

        self.PK1 = self.kinematics.Ft * self.Sigma
        self.sigma = (1./self.kinematics.Jt) * self.PK1 * dolfin.transpose(self.kinematics.Ft)

        # self.add_foi(expr=self.Sigma, fs=self.mfoi_fs, name="Sigma")
        # self.add_foi(expr=self.PK1  , fs=self.mfoi_fs, name="PK1"  )
        self.add_foi(expr=self.sigma, fs=self.mfoi_fs, name="sigma")



    def set_variational_formulation(self,
            penalties=[],
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

        self.res_form = dolfin.inner(
            self.sigma,
            dolfin.sym(dolfin.grad(self.subsols["U"].dsubtest))) * self.dV

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
