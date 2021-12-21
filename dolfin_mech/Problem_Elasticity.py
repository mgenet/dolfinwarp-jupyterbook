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
from .Problem import Problem

################################################################################

class ElasticityProblem(Problem):



    def __init__(self,
            w_incompressibility=False,
            mesh=None,
            compute_normals=False,
            domains_mf=None,
            boundaries_mf=None,
            points_mf=None,
            U_degree=None,
            p_degree=None,
            quadrature_degree=None,
            foi_degree=0,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None):

        Problem.__init__(self)

        self.w_incompressibility = w_incompressibility

        if (mesh is not None):
            self.set_mesh(
                mesh=mesh,
                compute_normals=compute_normals)

            self.set_measures(
                domains=domains_mf,
                boundaries=boundaries_mf,
                points=points_mf)

            self.set_subsols(
                U_degree=U_degree,
                p_degree=p_degree)
            self.set_solution_finite_element()
            self.set_solution_function_space()
            self.set_solution_functions()

            self.set_quadrature_degree(
                quadrature_degree=quadrature_degree)

            self.set_foi_finite_elements_DG(
                degree=foi_degree)
            self.set_foi_function_spaces()

            self.set_kinematics()

            self.add_elasticity_operators(
                elastic_behavior=elastic_behavior,
                elastic_behavior_dev=elastic_behavior_dev,
                elastic_behavior_bulk=elastic_behavior_bulk)



    def add_displacement_subsol(self,
            degree):

        self.U_degree = degree
        self.add_vector_subsol(
            name="u",
            family="CG",
            degree=self.U_degree)



    def get_displacement_subsol(self):

        return self.get_subsol("u")



    def add_pressure_subsol(self,
            degree):

        self.p_degree = degree
        if (self.p_degree == 0):
            self.add_scalar_subsol(
                name="p",
                family="DG",
                degree=self.p_degree)
        else:
            self.add_scalar_subsol(
                name="p",
                family="CG",
                degree=self.p_degree)



    def get_pressure_subsol(self):

        assert (self.w_incompressibility),\
            "There is no pressure subsol. Aborting."
        return self.get_subsol("p")



    def set_subsols(self,
            U_degree=1,
            p_degree=None):

        self.add_displacement_subsol(
            degree=U_degree)

        if (self.w_incompressibility):
            if (p_degree is None):
                p_degree = U_degree-1
            self.add_pressure_subsol(
                degree=p_degree)



    def get_displacement_function_space(self):

        if (len(self.subsols) == 1):
            return self.sol_fs
        else:
            return self.get_subsol_function_space(name="u")



    def get_pressure_function_space(self):

        assert (self.w_incompressibility),\
            "There is no pressure function space. Aborting."
        return self.get_subsol_function_space(name="p")



    def set_quadrature_degree(self,
            quadrature_degree=None):

        if (quadrature_degree is None) or (type(quadrature_degree) == int):
            pass
        elif (quadrature_degree == "full"):
            quadrature_degree = None
        elif (quadrature_degree == "default"):
            if   (self.mesh.ufl_cell().cellname() in ("triangle", "tetrahedron")):
                quadrature_degree = max(2, 2*(self.U_degree-1)) # MG20211221: This does not allow to reproduce full integration results exactly, but it is quite close…
            elif (self.mesh.ufl_cell().cellname() in ("quadrilateral", "hexahedron")):
                quadrature_degree = max(2, 2*(self.dim*self.U_degree-1))
        else:
            assert (0),\
                "Must provide an int, \"full\", \"default\" or None. Aborting."

        Problem.set_quadrature_degree(self,
            quadrature_degree=quadrature_degree)



    def set_kinematics(self):

        self.kinematics = dmech.LinearizedKinematics(
            dim=self.dim,
            U=self.get_displacement_subsol().subfunc,
            U_old=self.get_displacement_subsol().func_old)

        self.add_foi(expr=self.kinematics.epsilon, fs=self.mfoi_fs, name="epsilon")



    def add_elasticity_operator(self,
            elastic_behavior,
            subdomain_id=None):

        if (subdomain_id is None):
            measure = self.dV
        else:
            measure = self.dV(subdomain_id)
        operator = dmech.LinearizedElasticityOperator(
            u=self.get_displacement_subsol().subfunc,
            u_test=self.get_displacement_subsol().dsubtest,
            kinematics=self.kinematics,
            elastic_behavior=elastic_behavior,
            measure=measure)
        return self.add_operator(operator)



    def add_hydrostatic_pressure_operator(self,
            subdomain_id=None):

        if (subdomain_id is None):
            measure = self.dV
        else:
            measure = self.dV(subdomain_id)
        operator = dmech.LinearizedHydrostaticPressureOperator(
            u=self.get_displacement_subsol().subfunc,
            u_test=self.get_displacement_subsol().dsubtest,
            kinematics=self.kinematics,
            p=self.get_pressure_subsol().subfunc,
            measure=measure)
        return self.add_operator(operator)



    def add_incompressibility_operator(self,
            subdomain_id=None):

        if (subdomain_id is None):
            measure = self.dV
        else:
            measure = self.dV(subdomain_id)
        operator = dmech.LinearizedIncompressibilityOperator(
            kinematics=self.kinematics,
            p_test=self.get_pressure_subsol().dsubtest,
            measure=measure)
        return self.add_operator(operator)



    def add_elasticity_operators(self,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None):

        if (self.w_incompressibility):
            assert (elastic_behavior      is     None)
            assert (elastic_behavior_dev  is not None)
            assert (elastic_behavior_bulk is     None)

            if isinstance(elastic_behavior_dev, dmech.Material):
                operator = self.add_elasticity_operator(
                    elastic_behavior=elastic_behavior_dev)
                self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_dev")
                operator = self.add_hydrostatic_pressure_operator()
                operator = self.add_incompressibility_operator()
            elif type(elastic_behavior_dev) in (tuple,list):
                for (id,behavior) in elastic_behavior_dev:
                    operator = self.add_elasticity_operator(
                        elastic_behavior=behavior,
                        subdomain_id=id)
                    self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_dev")
                    operator = self.add_hydrostatic_pressure_operator()
                    operator = self.add_incompressibility_operator()
        else:
            if (elastic_behavior is not None):
                if isinstance(elastic_behavior, dmech.Material):
                    operator = self.add_elasticity_operator(
                        elastic_behavior=elastic_behavior)
                    self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma")
                elif type(elastic_behavior) in (tuple,list):
                    for (id,behavior) in elastic_behavior:
                        operator = self.add_elasticity_operator(
                            elastic_behavior=behavior,
                            subdomain_id=id)
                        self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma")
            elif ((elastic_behavior_dev  is not None)
              and (elastic_behavior_bulk is not None)):
                if  isinstance(elastic_behavior_dev , dmech.Material)\
                and isinstance(elastic_behavior_bulk, dmech.Material):
                    operator = self.add_elasticity_operator(
                        elastic_behavior=elastic_behavior_dev)
                    self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_dev")
                    operator = self.add_elasticity_operator(
                        elastic_behavior=elastic_behavior_bulk)
                    self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_bulk")
                elif type(elastic_behavior_dev)  in (tuple,list)\
                 and type(elastic_behavior_bulk) in (tuple,list):
                    for (id_dev, behavior_dev, id_bulk, behavior_bulk) in zip(elastic_behavior_dev, elastic_behavior_bulk):
                        assert (id_dev == id_bulk)
                        operator = self.add_elasticity_operator(
                            elastic_behavior=behavior_dev,
                            subdomain_id=id_dev)
                        self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_dev")
                        operator = self.add_elasticity_operator(
                            elastic_behavior=behavior_bulk,
                            subdomain_id=id_bulk)
                        self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_bulk")
            else:
                assert (0),\
                    "Must provide elastic_behavior or elastic_behavior_dev & elastic_behavior_bulk. Aborting."



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

        basename = "s_"

        self.add_qoi(
            name=basename+"XX",
            expr=sum([operator.sigma[0,0]*operator.measure for operator in self.operators if hasattr(operator, "sigma")]))
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"YY",
                expr=sum([operator.sigma[1,1]*operator.measure for operator in self.operators if hasattr(operator, "sigma")]))
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"ZZ",
                    expr=sum([operator.sigma[2,2]*operator.measure for operator in self.operators if hasattr(operator, "sigma")]))
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"XY",
                expr=sum([operator.sigma[0,1]*operator.measure for operator in self.operators if hasattr(operator, "sigma")]))
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"YZ",
                    expr=sum([operator.sigma[1,2]*operator.measure for operator in self.operators if hasattr(operator, "sigma")]))
                self.add_qoi(
                    name=basename+"ZX",
                    expr=sum([operator.sigma[2,0]*operator.measure for operator in self.operators if hasattr(operator, "sigma")]))



    def add_global_pressure_qoi(self):

        self.add_qoi(
            name="p",
            expr=sum([-dolfin.tr(operator.sigma)/3*operator.measure for operator in self.operators if hasattr(operator, "sigma")])+sum([operator.p*operator.measure for operator in self.operators if hasattr(operator, "p")]))
