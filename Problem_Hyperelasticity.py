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

class HyperelasticityProblem(Problem):



    def __init__(self,
            w_incompressibility=False,
            w_growth=False,
            w_relaxation=False,
            w_unloaded_configuration=False):

        Problem.__init__(self)

        self.w_incompressibility = w_incompressibility
        self.w_growth = w_growth
        self.w_relaxation = w_relaxation
        self.w_unloaded_configuration = w_unloaded_configuration



    def add_displacement_subsol(self,
            degree):

        self.add_vector_subsol(
            name="U",
            family="CG",
            degree=degree)



    def add_unloaded_displacement_subsol(self,
            degree):

        self.add_vector_subsol(
            name="Up",
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



    def add_unloaded_pressure_subsol(self,
            degree):

        if (degree == 0):
            self.add_scalar_subsol(
                name="Pp",
                family="DG",
                degree=0)
        else:
            self.add_scalar_subsol(
                name="Pp",
                family="CG",
                degree=degree)



    def add_growth_subsol(self,
            degree):

        self.add_scalar_subsol(
            name="thetag",
            family="DG",
            degree=degree)
        #self.add_tensor_subsol(
            #name="Fg",
            #family="DG",
            #degree=degree,
            #init_val=numpy.eye(self.dim))



    def add_relaxation_subsol(self,
            degree):

        self.add_tensor_subsol(
            name="Fr",
            family="DG",
            degree=degree,
            init_val=numpy.eye(self.dim))



    def set_subsols(self,
            U_degree=1):

        self.add_displacement_subsol(
            degree=U_degree)

        if (self.w_incompressibility):
            self.add_pressure_subsol(
                degree=U_degree-1)

        if (self.w_unloaded_configuration):
            self.add_unloaded_displacement_subsol(
                degree=U_degree)

            if (self.w_incompressibility):
                self.add_unloaded_pressure_subsol(
                    degree=U_degree-1)

        if (self.w_growth == "mixed"):
            self.add_growth_subsol(
                degree=U_degree-1)

        if (self.w_relaxation == "mixed"):
            self.add_relaxation_subsol(
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



    def get_unloaded_displacement_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="Up")



    def get_pressure_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="P")



    def get_unloaded_pressure_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="Pp")



    def get_growth_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="thetag")
        #return self.get_subsol_function_space(name="Fg")



    def get_relaxation_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="Fr")



    def set_materials(self,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None,
            growth_behavior=None,
            relaxation_behavior=None):

        if (elastic_behavior is not None):
            self.elastic_behavior = elastic_behavior
        else:
            assert (elastic_behavior_dev is not None)
            self.elastic_behavior_dev = elastic_behavior_dev
            if not (self.w_incompressibility):
                assert (elastic_behavior_bulk is not None)
                self.elastic_behavior_bulk = elastic_behavior_bulk

        if (self.w_growth):
            assert (growth_behavior is not None)
            if (self.w_growth == "mixed"):
                self.inelastic_behaviors_mixed    += [growth_behavior]
            elif (self.w_growth == "internal"):
                self.inelastic_behaviors_internal += [growth_behavior]
        if (self.w_relaxation):
            assert (relaxation_behavior is not None)
            if (self.w_relaxation == "mixed"):
                self.inelastic_behaviors_mixed    += [relaxation_behavior]
            elif (self.w_relaxation == "internal"):
                self.inelastic_behaviors_internal += [relaxation_behavior]

        for inelastic_behavior in self.inelastic_behaviors_mixed:
            inelastic_behavior.set_internal_variables_mixed(
                problem=self)
        for inelastic_behavior in self.inelastic_behaviors_internal:
            inelastic_behavior.set_internal_variables_internal(
                problem=self)

        self.kinematics = dcm.Kinematics(
            dim=self.dim,
            U=self.subsols["U"].subfunc,
            U_old=self.subsols["U"].func_old,
            Q_expr=self.Q_expr,
            w_growth=self.w_growth,
            Fg=growth_behavior.Fg if (self.w_growth) else None,
            Fg_old=growth_behavior.Fg_old if (self.w_growth) else None,
            w_relaxation=self.w_relaxation,
            Fr=relaxation_behavior.Fr if (self.w_relaxation) else None,
            Fr_old=relaxation_behavior.Fr_old if (self.w_relaxation) else None)

        if (self.w_growth) or (self.w_relaxation):
            self.add_foi(expr=self.kinematics.Ft, fs=self.mfoi_fs, name="Ft")
            self.add_foi(expr=self.kinematics.Jt, fs=self.sfoi_fs, name="Jt")
            self.add_foi(expr=self.kinematics.Ct, fs=self.mfoi_fs, name="Ct")
            self.add_foi(expr=self.kinematics.Et, fs=self.mfoi_fs, name="Et")
            self.add_foi(expr=self.kinematics.Fe, fs=self.mfoi_fs, name="Fe")
            self.add_foi(expr=self.kinematics.Je, fs=self.sfoi_fs, name="Je")
            self.add_foi(expr=self.kinematics.Ce, fs=self.mfoi_fs, name="Ce")
            self.add_foi(expr=self.kinematics.Ee, fs=self.mfoi_fs, name="Ee")
            if (self.Q_expr is not None):
                self.add_foi(expr=self.kinematics.Ee_loc, fs=self.mfoi_fs, name="Ee_loc")
        else:
            self.add_foi(expr=self.kinematics.Fe, fs=self.mfoi_fs, name="F")
            self.add_foi(expr=self.kinematics.Je, fs=self.sfoi_fs, name="J")
            self.add_foi(expr=self.kinematics.Ce, fs=self.mfoi_fs, name="C")
            self.add_foi(expr=self.kinematics.Ee, fs=self.mfoi_fs, name="E")
            if (self.Q_expr is not None):
                self.add_foi(expr=self.kinematics.Ee_loc, fs=self.mfoi_fs, name="Ee_loc")

        if (elastic_behavior is not None):
            self.Psi, self.Sigma = self.elastic_behavior.get_free_energy(
                C=self.kinematics.Ce)
            self.Psi = [self.Psi]
            self.Sigma = [self.Sigma]
        else:
            if (self.w_incompressibility):
                self.Psi_bulk   = -self.subsols["P"].subfunc * (self.kinematics.Je - 1)
                self.Sigma_bulk = -self.subsols["P"].subfunc *  self.kinematics.Je      * self.kinematics.Ce_inv
            else:
                self.Psi_bulk, self.Sigma_bulk = self.elastic_behavior_bulk.get_free_energy(
                    C=self.kinematics.Ce)
            self.Psi_dev, self.Sigma_dev = self.elastic_behavior_dev.get_free_energy(
                C=self.kinematics.Ce)
        # self.Psi_dev, self.Sigma_dev = self.elastic_behavior_dev.get_free_energy(
        #     C=self.kinematics.Ce)
        if elastic_behavior_dev and elastic_behavior_bulk:
            self.Psi   = [Psi_bulk_elt + Psi_dev_elt for Psi_bulk_elt, Psi_dev_elt in zip(self.Psi_bulk, self.Psi_dev)]
            self.Sigma = [Sigma_bulk_elt + Sigma_dev_elt for Sigma_bulk_elt, Sigma_dev_elt in zip(self.Sigma_bulk, self.Sigma_dev)]

        #self.kinematics.Ee = dolfin.variable(self.kinematics.Ee) # MG20180412: Works here,
        #self.Sigma = dolfin.diff(self.Psi, self.kinematics.Ee)   # MG20180412: but fails at project…

        self.PK1 = [self.kinematics.Ft * self.Sigma[k] for k in range(len(self.Sigma))]
        self.sigma = [(1./self.kinematics.Jt) * self.PK1[k] * dolfin.transpose(self.kinematics.Ft) for k in range(len(self.PK1))]

        # self.add_foi(expr=self.Sigma, fs=self.mfoi_fs, name="Sigma")
        # #self.add_foi(expr=self.PK1  , fs=self.mfoi_fs, name="PK1"  )
        # self.add_foi(expr=self.sigma, fs=self.mfoi_fs, name="sigma")

        if (self.Q_expr is not None):
            self.sigma_loc = dolfin.dot(dolfin.dot(self.Q_expr, self.sigma), dolfin.transpose(self.Q_expr))
            self.add_foi(expr=self.sigma_loc, fs=self.mfoi_fs, name="sigma_loc")

        if (self.w_unloaded_configuration):
            self.unloaded_kinematics = dcm.Kinematics(
                dim=self.dim,
                U=self.subsols["Up"].subfunc,
                U_old=self.subsols["Up"].func_old,
                Q_expr=self.Q_expr,
                w_growth=self.w_growth,
                Fg=growth_behavior.Fg if (self.w_growth) else None,
                Fg_old=growth_behavior.Fg_old if (self.w_growth) else None,
                w_relaxation=self.w_relaxation,
                Fr=relaxation_behavior.Fr if (self.w_relaxation) else None,
                Fr_old=relaxation_behavior.Fr_old if (self.w_relaxation) else None)

            self.add_foi(expr=self.unloaded_kinematics.Ft, fs=self.mfoi_fs, name="Ftp")
            self.add_foi(expr=self.unloaded_kinematics.Jt, fs=self.sfoi_fs, name="Jtp")
            self.add_foi(expr=self.unloaded_kinematics.Ct, fs=self.mfoi_fs, name="Ctp")
            self.add_foi(expr=self.unloaded_kinematics.Et, fs=self.mfoi_fs, name="Etp")
            self.add_foi(expr=self.unloaded_kinematics.Fe, fs=self.mfoi_fs, name="Fep")
            self.add_foi(expr=self.unloaded_kinematics.Je, fs=self.sfoi_fs, name="Jep")
            self.add_foi(expr=self.unloaded_kinematics.Ce, fs=self.mfoi_fs, name="Cep")
            self.add_foi(expr=self.unloaded_kinematics.Ee, fs=self.mfoi_fs, name="Eep")
            if (self.Q_expr is not None):
                self.add_foi(expr=self.unloaded_kinematics.Ee_loc, fs=self.mfoi_fs, name="Eep_loc")

            if (elastic_behavior is not None):
                self.unloaded_Psi, self.unloaded_Sigma = self.elastic_behavior.get_free_energy(
                    C=self.unloaded_kinematics.Ce)
            else:
                if (self.w_incompressibility):
                    self.unloaded_Psi_bulk   = -self.subsols["Pp"].subfunc * (self.unloaded_kinematics.Je - 1)
                    self.unloaded_Sigma_bulk = -self.subsols["Pp"].subfunc *  self.unloaded_kinematics.Je      * self.unloaded_kinematics.Ce_inv
                else:
                    self.unloaded_Psi_bulk, self.unloaded_Sigma_bulk = self.elastic_behavior_bulk.get_free_energy(
                        C=self.unloaded_kinematics.Ce)
                self.unloaded_Psi_dev, self.unloaded_Sigma_dev = self.elastic_behavior_dev.get_free_energy(
                    C=self.unloaded_kinematics.Ce)
                self.unloaded_Psi   = self.unloaded_Psi_bulk   + self.unloaded_Psi_dev
                self.unloaded_Sigma = self.unloaded_Sigma_bulk + self.unloaded_Sigma_dev

            #self.unloaded_kinematics.Ee = dolfin.variable(self.unloaded_kinematics.Ee)        # MG20180412: Works here,
            #self.unloaded_Sigma = dolfin.diff(self.unloaded_Psi, self.unloaded_kinematics.Ee) # MG20180412: but fails at project…

            self.unloaded_PK1 = self.unloaded_kinematics.Ft * self.unloaded_Sigma
            self.unloaded_sigma = (1./self.unloaded_kinematics.Jt) * self.unloaded_PK1 * dolfin.transpose(self.unloaded_kinematics.Ft)

            self.add_foi(expr=self.unloaded_Sigma, fs=self.mfoi_fs, name="Sigmap")
            self.add_foi(expr=self.unloaded_PK1  , fs=self.mfoi_fs, name="PK1p"  )
            self.add_foi(expr=self.unloaded_sigma, fs=self.mfoi_fs, name="sigmap")

            if (self.Q_expr is not None):
                self.unloaded_sigma_loc = dolfin.dot(dolfin.dot(self.Q_expr, self.unloaded_sigma), dolfin.transpose(self.Q_expr))
                self.add_foi(expr=self.unloaded_sigma_loc, fs=self.mfoi_fs, name="sigmap_loc")



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
            nb_material_subdomains=1,
            dt=None):

        if nb_material_subdomains == 1:
            assert len(self.Psi) == 1
            self.Pi = self.Psi[0] * self.dV
            # print self.Pi
        else:
            # self.Pi = self.Psi[0] * self.dV(0) + self.Psi[1] * self.dV(1)
            self.Pi = self.Psi[0] * self.dV(0)
            for id_subdomain in range(1, nb_material_subdomains):
                self.Pi += self.Psi[id_subdomain] * self.dV(id_subdomain)

        if (self.w_unloaded_configuration):
            self.Pi += self.unloaded_Psi * self.dV

        for loading in normal_penalties:
            self.Pi += (loading.val/2) * dolfin.inner(
                self.subsols["U"].subfunc,
                self.mesh_normals)**2 * loading.measure

        if (self.w_unloaded_configuration):
            for loading in normal_penalties:
                self.Pi += (loading.val/2) * dolfin.inner(
                    self.subsols["Up"].subfunc,
                    self.mesh_normals)**2 * loading.measure

        # for loading in directional_penalties: #MG20190513: Cannot use point integral within assemble_system
        #     self.Pi += (loading.val/2) * dolfin.inner(
        #         self.subsols["U"].subfunc,
        #         loading.N)**2 * loading.measure
        #
        # if (self.w_unloaded_configuration):
        #     for loading in directional_penalties: #MG20190513: Cannot use point integral within assemble_system
        #         self.Pi += (loading.val/2) * dolfin.inner(
        #             self.subsols["Up"].subfunc,
        #             loading.N)**2 * loading.measure

        for loading in surface_tensions:
            FmTN = dolfin.dot(
                dolfin.transpose(dolfin.inv(self.kinematics.Ft)),
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
            self.dsol_test); assert (self.w_growth != "mixed") and (self.w_relaxation != "mixed")

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

        # if (self.w_unloaded_configuration):
        #     self.res_form += dolfin.inner(
        #         self.unloaded_Sigma,
        #         dolfin.derivative(
        #             self.unloaded_kinematics.Et,
        #             self.subsols["Up"].subfunc,
        #             self.subsols["Up"].dsubtest)) * self.dV
        #
        #     if (self.w_incompressibility):
        #         self.res_form += dolfin.inner(
        #             self.unloaded_kinematics.Je-1,
        #             self.subsols["Pp"].dsubtest) * self.dV

        # for loading in normal_penalties:
        #     self.res_form += loading.val * dolfin.inner(
        #         self.subsols["U"].subfunc,
        #         self.mesh_normals) * dolfin.inner(
        #         self.subsols["U"].dsubtest,
        #         self.mesh_normals) * loading.measure
        #
        # if (self.w_unloaded_configuration):
        #     for loading in normal_penalties:
        #         self.res_form += loading.val * dolfin.inner(
        #             self.subsols["Up"].subfunc,
        #             self.mesh_normals) * dolfin.inner(
        #             self.subsols["Up"].dsubtest,
        #             self.mesh_normals) * loading.measure

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
                dolfin.transpose(dolfin.inv(self.kinematics.Ft)),
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

        for inelastic_behavior in self.inelastic_behaviors_mixed:
            self.res_form += inelastic_behavior.get_res_term(
                problem=self,
                dt=dt)

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)

        print self.jac_form



    def add_strain_qois(self,
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



    def add_stress_qois(self,
            stress_type="cauchy"):

        if (stress_type in ("cauchy", "sigma")):
            basename = "s_"
            stress = self.sigma
        elif (stress_type in ("piola", "PK2", "Sigma")):
            basename = "S_"
            stress = self.Sigma
        elif (stress_type in ("PK1", "P")):
            basename = "P_"
            stress = self.PK1

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
