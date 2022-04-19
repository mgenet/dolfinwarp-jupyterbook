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
# from .Problem_Hyperelasticity import HyperelasticityProblem

################################################################################

class MicroHyperelasticityProblem(Problem):



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
                define_spatial_coordinates=1,
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



    def add_perturbation_subsol(self,
            degree):

        self.U_degree = degree
        self.add_vector_subsol(
            name="U_tilde",
            family="CG",
            degree=self.U_degree)



    def get_perturbation_subsol(self):

        return self.get_subsol("U_tilde")



    def add_pressure_subsol(self,
            degree):

        self.p_degree = degree
        if (degree == 0):
            self.add_scalar_subsol(
                name="P",
                family="DG",
                degree=0)
        else:
            self.add_scalar_subsol(
                name="P",
                family="CG",
                degree=self.p_degree)



    def get_pressure_subsol(self):

        assert (self.w_incompressibility),\
            "There is no pressure subsol. Aborting."
        return self.get_subsol("P")



    def set_subsols(self,
            U_degree=None,
            p_degree=None):

        self.add_perturbation_subsol(
            degree=U_degree)

        if (self.w_incompressibility):
            if (p_degree is None):
                p_degree = U_degree-1
            self.add_pressure_subsol(
                degree=p_degree)



    def get_perturbation_function_space(self):

        if (len(self.subsols) == 1):
            return self.sol_fs
        else:
            return self.get_subsol_function_space(name="U_tilde")



    def get_pressure_function_space(self):

        assert (self.w_incompressibility),\
            "There is no pressure function space. Aborting."
        return self.get_subsol_function_space(name="P")



    def set_quadrature_degree(self,
            quadrature_degree=None):

        if (quadrature_degree is None) or (type(quadrature_degree) == int):
            pass
        elif (quadrature_degree == "full"):
            quadrature_degree = None
        elif (quadrature_degree == "default"):
            if   (self.mesh.ufl_cell().cellname() in ("triangle", "tetrahedron")):
                quadrature_degree = max(2, 4*(self.U_degree-1)) # MG20211221: This does not allow to reproduce full integration results exactly, but it is quite close…
            elif (self.mesh.ufl_cell().cellname() in ("quadrilateral", "hexahedron")):
                quadrature_degree = max(2, 4*(self.dim*self.U_degree-1))
        else:
            assert (0),\
                "Must provide an int, \"full\", \"default\" or None. Aborting."

        Problem.set_quadrature_degree(self,
            quadrature_degree=quadrature_degree)



    def set_kinematics(self):

        self.kinematics = dmech.Kinematics(
            dim=self.dim,
            # F_bar=self.F_bar,
            U=self.U_bar+self.get_perturbation_subsol().subfunc,
            U_old=self.U_bar+self.get_perturbation_subsol().func_old,
            Q_expr=self.Q_expr)

        self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F")
        self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J")
        self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C")
        self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E")
        if (self.Q_expr is not None):
            self.add_foi(expr=self.kinematics.E_loc, fs=self.mfoi_fs, name="E_loc")



    def add_elasticity_operator(self,
            elastic_behavior,
            subdomain_id=None):

        if (subdomain_id is None):
            measure = self.dV
        else:
            measure = self.dV(subdomain_id)
        operator = dmech.MicroHyperElasticityOperator(
            U=self.U_bar + self.get_perturbation_subsol().subfunc,
            U_test=self.get_perturbation_subsol().dsubtest,
            kinematics=self.kinematics,
            elastic_behavior=elastic_behavior,
            measure=measure)


        # U=self.get_perturbation_subsol().subfunc
        # U_test=self.get_perturbation_subsol().dsubtest
        # Psi, self.Sigma = elastic_behavior.get_free_energy(
        #     C=self.kinematics.C)
        # self.PK1   = self.kinematics.F * self.Sigma
        # self.sigma = self.PK1 * self.kinematics.F.T / self.kinematics.J

        # self.measure = self.dV(subdomain_id)

        # # self.res_form = dolfin.derivative(Psi, U, U_test) * self.measure

        # # dE_test = dolfin.derivative(
        # #     Psi, U, U_test)

        # self.res_form = dolfin.inner(self.Sigma, dolfin.nabla_grad(U_test)) * self.measure

        return self.add_operator(operator)

    def add_pressure_loading_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.PressureLoadingOperator(
            U_test=self.get_perturbation_subsol().dsubtest,
            kinematics=self.kinematics,
            N=self.mesh_normals,
            **kwargs)
        self.add_operator(
            operator=operator,
            k_step=k_step)
        return operator



    def add_hydrostatic_pressure_operator(self,
            subdomain_id=None):

        if (subdomain_id is None):
            measure = self.dV
        else:
            measure = self.dV(subdomain_id)
        operator = dmech.HyperHydrostaticPressureOperator(
            U=self.U_bar + self.get_perturbation_subsol().subfunc,
            U_test=self.get_perturbation_subsol().dsubtest,
            kinematics=self.kinematics,
            P=self.get_pressure_subsol().subfunc,
            measure=measure)
        return self.add_operator(operator)



    def add_incompressibility_operator(self,
            subdomain_id=None):

        if (subdomain_id is None):
            measure = self.dV
        else:
            measure = self.dV(subdomain_id)
        operator = dmech.HyperIncompressibilityOperator(
            kinematics=self.kinematics,
            P_test=self.get_pressure_subsol().dsubtest,
            measure=measure)
        return self.add_operator(operator)

##############################################################################################

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
                self.add_foi(expr=operator.Sigma, fs=self.mfoi_fs, name="Sigma_dev")
                self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_dev")
                operator = self.add_hydrostatic_pressure_operator()
                operator = self.add_incompressibility_operator()
            elif type(elastic_behavior_dev) in (tuple,list):
                for (id,behavior) in elastic_behavior_dev:
                    operator = self.add_elasticity_operator(
                        elastic_behavior=behavior,
                        subdomain_id=id)
                    self.add_foi(expr=operator.Sigma, fs=self.mfoi_fs, name="Sigma_dev")
                    self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_dev")
                    operator = self.add_hydrostatic_pressure_operator()
                    operator = self.add_incompressibility_operator()
        else:
            if (elastic_behavior is not None):
                if isinstance(elastic_behavior, dmech.Material):
                    operator = self.add_elasticity_operator(
                        elastic_behavior=elastic_behavior)
                    self.add_foi(expr=operator.Sigma, fs=self.mfoi_fs, name="Sigma")
                    self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma")
                    self.add_foi(expr=self.U_bar + self.get_perturbation_subsol().subfunc, fs=self.vfoi_fs, name="u_tot")
                    # self.add_foi(expr=self.U_bar, fs=self.vfoi_fs, name="u_tot")

                elif type(elastic_behavior) in (tuple,list):
                    for (id,behavior) in elastic_behavior:
                        operator = self.add_elasticity_operator(
                            elastic_behavior=behavior,
                            subdomain_id=id)
                        self.add_foi(expr=operator.Sigma, fs=self.mfoi_fs, name="Sigma")
                        self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma")
            elif ((elastic_behavior_dev  is not None)
              and (elastic_behavior_bulk is not None)):
                if  isinstance(elastic_behavior_dev , dmech.Material)\
                and isinstance(elastic_behavior_bulk, dmech.Material):
                    operator = self.add_elasticity_operator(
                        elastic_behavior=elastic_behavior_dev)
                    self.add_foi(expr=operator.Sigma, fs=self.mfoi_fs, name="Sigma_dev")
                    self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_dev")
                    operator = self.add_elasticity_operator(
                        elastic_behavior=elastic_behavior_bulk)
                    self.add_foi(expr=operator.Sigma, fs=self.mfoi_fs, name="Sigma_bulk")
                    self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_bulk")
                elif type(elastic_behavior_dev)  in (tuple,list)\
                 and type(elastic_behavior_bulk) in (tuple,list):
                    for (id_dev, behavior_dev, id_bulk, behavior_bulk) in zip(elastic_behavior_dev, elastic_behavior_bulk):
                        assert (id_dev == id_bulk)
                        operator = self.add_elasticity_operator(
                            elastic_behavior=behavior_dev,
                            subdomain_id=id_dev)
                        self.add_foi(expr=operator.Sigma, fs=self.mfoi_fs, name="Sigma_dev")
                        self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_dev")
                        operator = self.add_elasticity_operator(
                            elastic_behavior=behavior_bulk,
                            subdomain_id=id_bulk)
                        self.add_foi(expr=operator.Sigma, fs=self.mfoi_fs, name="Sigma_bulk")
                        self.add_foi(expr=operator.sigma, fs=self.mfoi_fs, name="sigma_bulk")
            else:
                assert (0),\
                    "Must provide elastic_behavior or elastic_behavior_dev & elastic_behavior_bulk. Aborting."
        
        if (self.Q_expr is not None):
            assert (0), "ToDo. Aborting."
            # operator.sigma_loc = dolfin.dot(dolfin.dot(self.Q_expr, operator.sigma), self.Q_expr.T)
            # self.add_foi(expr=operator.sigma_loc, fs=self.mfoi_fs, name="sigma_loc")



    def add_global_strain_qois(self):

        basename = "E_"
        strain = self.kinematics.E

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



    def add_global_volume_ratio_qois(self):

        basename = "J"
        J = self.kinematics.J

        self.add_qoi(
            name=basename,
            expr=J / self.mesh_V0 * self.dV)



    def add_global_stress_qois(self,
            stress_type="cauchy"):

        if (stress_type in ("Cauchy", "cauchy", "sigma")):
            basename = "s_"
            stress = "sigma"
        elif (stress_type in ("Piola", "piola", "PK2", "Sigma")):
            basename = "S_"
            stress = "Sigma"
        elif (stress_type in ("Boussinesq", "boussinesq", "PK1", "P")):
            basename = "P_"
            stress = "P"

        self.add_qoi(
            name=basename+"XX",
            expr=sum([getattr(operator, stress)[0,0]*operator.measure for operator in self.operators if hasattr(operator, stress)]))
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"YY",
                expr=sum([getattr(operator, stress)[1,1]*operator.measure for operator in self.operators if hasattr(operator, stress)]))
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"ZZ",
                    expr=sum([getattr(operator, stress)[2,2]*operator.measure for operator in self.operators if hasattr(operator, stress)]))
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"XY",
                expr=sum([getattr(operator, stress)[0,1]*operator.measure for operator in self.operators if hasattr(operator, stress)]))
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"YZ",
                    expr=sum([getattr(operator, stress)[1,2]*operator.measure for operator in self.operators if hasattr(operator, stress)]))
                self.add_qoi(
                    name=basename+"ZX",
                    expr=sum([getattr(operator, stress)[2,0]*operator.measure for operator in self.operators if hasattr(operator, stress)]))



    def add_global_pressure_qoi(self):

        self.add_qoi(
            name="P",
            expr=sum([-dolfin.tr(operator.sigma)/3*operator.measure for operator in self.operators if hasattr(operator, "sigma")])+sum([operator.P*operator.measure for operator in self.operators if hasattr(operator, "P")])) 
