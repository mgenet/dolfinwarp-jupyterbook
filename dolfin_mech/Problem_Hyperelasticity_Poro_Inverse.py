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

################################################################################

class InversePoroHyperelasticityProblem(PoroHyperelasticityProblem):



    def __init__(self, *args, **kwargs):

        PoroHyperelasticityProblem.__init__(self)



    def add_porosity_subsol(self,
            degree,
            init_val=None,
            init_fun=None):

        if (degree == 0):
            self.add_scalar_subsol(
                name="phis0",
                family="DG",
                degree=0,
                init_val=init_val,
                init_fun=init_fun)
        else:
            self.add_scalar_subsol(
                name="phis0",
                family="CG",
                degree=degree,
                init_val=init_val,
                init_fun=init_fun)



    def get_porosity_subsol(self):

        return self.get_subsol("phis0")



    def get_porosity_function_space(self):

        return self.get_subsol_function_space(name="phis0")



    def set_kinematics(self):

        self.kinematics = dmech.InverseKinematics(
            U=self.get_displacement_subsol().subfunc,
            U_old=self.get_displacement_subsol().func_old)

        self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F")
        self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J")
        self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C")
        self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E")



    def add_Wskel_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.WskelPoroOperator(
            kinematics=self.kinematics,
            Phis0=self.Phis0,
            U_test=self.get_displacement_subsol().dsubtest,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_Wskel_operators(self,
            skel_behaviors):

        for skel_behavior in skel_behaviors:
            operator = self.add_Wskel_operator(
                material_parameters=skel_behavior["parameters"],
                material_scaling=skel_behavior["scaling"],
                subdomain_id=skel_behavior.get("subdomain_id", None))
            suffix = "_"+skel_behavior["suffix"] if "suffix" in skel_behavior else ""
            self.add_foi(expr=operator.material.Sigma, fs=self.mfoi_fs, name="Sigma_skel"+suffix)
            self.add_foi(expr=operator.material.sigma, fs=self.mfoi_fs, name="sigma_skel"+suffix)



    def add_Wbulk_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.WbulkPoroOperator(
            kinematics=self.kinematics,
            Phis0=self.Phis0,
            Phis=self.get_porosity_subsol().subfunc,
            U_test=self.get_displacement_subsol().dsubtest,
            Phis_test=self.get_porosity_subsol().dsubtest,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_Wbulk_operators(self,
            bulk_behaviors):

        for bulk_behavior in bulk_behaviors:
            operator = self.add_Wbulk_operator(
                material_parameters=bulk_behavior["parameters"],
                material_scaling=bulk_behavior["scaling"],
                subdomain_id=bulk_behavior.get("subdomain_id", None))
            suffix = "_"+bulk_behavior["suffix"] if "suffix" in bulk_behavior else ""
            self.add_foi(expr=operator.material.dWbulkdPhis, fs=self.sfoi_fs, name="dWbulkdPhis"+suffix)
            self.add_foi(expr=operator.material.dWbulkdPhis * self.kinematics.J * self.kinematics.C_inv, fs=self.mfoi_fs, name="Sigma_bulk"+suffix)
            self.add_foi(expr=operator.material.dWbulkdPhis * self.kinematics.I, fs=self.mfoi_fs, name="sigma_bulk"+suffix)



    def add_Wpore_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.WporePoroOperator(
            kinematics=self.kinematics,
            U_test=self.get_displacement_subsol().dsubtest,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_Wpore_operators(self,
            pore_behaviors):

        for pore_behavior in pore_behaviors:
            self.add_Wpore_operator(
                material_parameters=pore_behavior["parameters"],
                material_scaling=pore_behavior["scaling"],
                subdomain_id=pore_behavior.get("subdomain_id", None))



    def add_pf_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.PfPoroOperator(
            Phis_test=self.get_porosity_subsol().dsubtest,
            **kwargs)
        self.add_operator(
            operator=operator,
            k_step=k_step)
        self.add_foi(expr=operator.pf, fs=self.sfoi_fs, name="pf")



    def add_global_porosity_qois(self):

        self.add_qoi(
            name="Phis",
            expr=self.get_porosity_subsol().subfunc * self.dV)

        self.add_qoi(
            name="Phif",
            expr=(self.kinematics.J - self.get_porosity_subsol().subfunc) * self.dV)



    def add_global_stress_qois(self,
            stress_type="cauchy"):

        if (stress_type in ("Cauchy", "cauchy", "sigma")):
            basename = "s_"
            stress = "sigma"
        elif (stress_type in ("Piola", "piola", "PK2", "Sigma")):
            basename = "S_"
            stress = "Sigma"
        elif (stress_type in ("Boussinesq", "boussinesq", "PK1", "P")):
            assert (0), "ToDo. Aborting."

        compnames = ["XX"]
        comps     = [(0,0)]
        if (self.dim >= 2):
            compnames += ["YY"]
            comps     += [(1,1)]
            if (self.dim >= 3):
                compnames += ["ZZ"]
                comps     += [(2,2)]
            compnames += ["XY"]
            comps     += [(0,1)]
            if (self.dim >= 3):
                compnames += ["YZ"]
                comps     += [(1,2)]
                compnames += ["ZX"]
                comps     += [(2,0)]
        for compname, comp in zip(compnames, comps):
            if (stress == "Sigma"):
                self.add_qoi(
                    name=basename+"skel_"+compname,
                    expr=sum([getattr(operator.material, stress)[comp]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))
                self.add_qoi(
                    name=basename+"bulk_"+compname,
                    expr=sum([getattr(operator.material, "dWbulkdPhis")*self.kinematics.J*self.kinematics.C_inv[comp]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, "dWbulkdPhis"))]))
                self.add_qoi(
                    name=basename+"tot_"+compname,
                    expr=sum([getattr(operator.material, stress)[comp]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))+sum([getattr(operator.material, "dWbulkdPhis")[comp]*self.kinematics.J*self.kinematics.C_inv*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, "dWbulkdPhis"))])
            elif (stress == "sigma"):
                self.add_qoi(
                    name=basename+"skel_"+compname,
                    expr=sum([getattr(operator.material, stress)[comp]*self.kinematics.J*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))
                self.add_qoi(
                    name=basename+"bulk_"+compname,
                    expr=sum([getattr(operator.material, "dWbulkdPhis")*self.kinematics.I[comp]*self.kinematics.J*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, "dWbulkdPhis"))]))
                self.add_qoi(
                    name=basename+"tot_"+compname,
                    expr=sum([getattr(operator.material, stress)[comp]*self.kinematics.J*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))])+sum([getattr(operator.material, "dWbulkdPhis")*self.kinematics.I[comp]*self.kinematics.J*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, "dWbulkdPhis"))]))



    def add_global_fluid_pressure_qoi(self):

        # for operator in self.operators:
        #     print(type(operator))
        #     print(hasattr(operator, "pf"))

        # for step in self.steps:
        #     print (step)
        #     for operator in step.operators:
        #         print(type(operator))
        #         print(hasattr(operator, "pf"))

        self.add_qoi(
            name="pf",
            expr=sum([operator.pf*operator.measure for step in self.steps for operator in step.operators if hasattr(operator, "pf")]))
