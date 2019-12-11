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
from .Problem_InverseHyperelasticity import InverseHyperelasticityProblem

################################################################################

class TwoSubfuncInversePoroProblem(InverseHyperelasticityProblem):



    def __init__(self,
            eta,
            kappa,
            p0 = 0):

        InverseHyperelasticityProblem.__init__(self,w_incompressibility=False)
        self.eta   = eta
        self.kappa = kappa
        self.p0    = p0



    def add_porosity_subsol(self,
            degree):

        if (degree == 0):
            self.add_scalar_subsol(
                name="Phi0",
                family="DG",
                degree=0,
                init_val=numpy.array([self.value_phi_given]))
                # init_val=self.porosity_given)
        else:
            self.add_scalar_subsol(
                name="Phi0",
                family="CG",
                degree=degree,
                init_val=numpy.array([self.value_phi_given]))
                # init_val=self.porosity_given)



    def set_subsols(self,
            U_degree=1):

        self.add_displacement_subsol(
            degree=U_degree)

        self.add_porosity_subsol(
            degree=U_degree-1)



    def get_porosity_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="Phi0")



    def set_porosity_energy(self):

        # exp
        dWpordJs = self.eta * (self.Phi0 / (self.kinematics.Je * self.Phi))**2 * dolfin.exp(-self.kinematics.Je * self.Phi / (self.Phi0 - self.kinematics.Je * self.Phi)) / (self.Phi0 - self.kinematics.Je * self.Phi)
        # n = 2
        # dWpordJs = self.eta * n * ((self.Phi0 - self.kinematics.Je * self.Phi) / (self.kinematics.Je * self.Phi))**(n-1) * self.Phi0 / (self.kinematics.Je * self.Phi)**2
        # dWpordJs = 0
        dWpordJs_condition = dolfin.conditional(dolfin.lt(self.Phi, self.Phi0), dWpordJs, 0)
        self.dWpordJs = (1 - self.Phi0) * dWpordJs_condition

        # n = 4
        # dWpordJs = - self.eta * n * ((self.Phi0 - self.kinematics.Je * self.Phi) / (self.kinematics.Je * self.Phi*self.Phi0))**(n-1) / (self.kinematics.Je * self.Phi)**2
        # dWpordJs_condition = dolfin.conditional(dolfin.lt(self.Phi0, self.Phi), dWpordJs, 0)
        # self.dWpordJs = (1 - self.Phi0) * dWpordJs_condition



    def set_bulk_energy(self):

        self.Wbulk = self.kappa * (self.kinematics.Js / (1. - self.Phi0) - 1 - dolfin.ln(self.kinematics.Js / (1. - self.Phi0)))
        dWbulkdJs = self.kappa * (1. / (1. - self.Phi0) - 1./self.kinematics.Js)
        self.dWbulkdJs = (1 - self.Phi0) * dWbulkdJs



    def set_Phi0_and_Phi(self,
            config_porosity='ref'):

        if self.config_porosity == 'ref':
            self.Phi0 = Nan
            self.Phi  = Nan
        elif self.config_porosity == 'deformed':
            self.Phi0 = self.subsols["Phi0"].subfunc
            # self.Phi0 = dolfin.conditional(dolfin.gt(self.Phi0,0), self.Phi0, 0)
            self.Phi0bin = dolfin.conditional(dolfin.gt(self.Phi0,0), 1, 0)
            self.Phi  = self.porosity_given



    def set_kinematics(self):

        InverseHyperelasticityProblem.set_kinematics(self)

        self.set_Phi0_and_Phi(self.config_porosity)
        self.kinematics.Js = self.kinematics.Je * (1 - self.Phi)



    def set_materials(self,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None,
            subdomain_id=None):

        self.set_kinematics()

        InverseHyperelasticityProblem.set_materials(self,
                elastic_behavior=elastic_behavior,
                elastic_behavior_dev=elastic_behavior_dev,
                elastic_behavior_bulk=elastic_behavior_bulk,
                subdomain_id=subdomain_id)

        self.set_porosity_energy()
        self.set_bulk_energy()



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

        self.res_form = 0

        for subdomain in self.subdomains :
            self.res_form += dolfin.inner(
                subdomain.sigma,
                dolfin.sym(dolfin.grad(self.subsols["U"].dsubtest))) * self.dV(subdomain.id)

        for loading in pressure_loadings:
            self.res_form -= dolfin.inner(
               -loading.val * self.mesh_normals,
                self.subsols["U"].dsubtest) * loading.measure

        self.res_form += dolfin.inner(
            self.dWbulkdJs * self.kinematics.Je * self.kinematics.Ce_inv,
            # - (self.p0 + self.dWpordJs) * self.kinematics.Je * self.kinematics.Ce_inv,
            dolfin.derivative(
                    self.kinematics.Et,
                    self.subsols["U"].subfunc,
                    self.subsols["U"].dsubtest)) * self.dV

        self.res_form += dolfin.inner(
                self.Phi0bin * (self.dWbulkdJs + self.dWpordJs + self.p0) + (1 - self.Phi0bin) * self.subsols["Phi0"].subfunc,
                self.subsols["Phi0"].dsubtest) * self.dV

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)



    def add_Phi0_qois(self):

        basename = "PHI0_"

        self.add_qoi(
            name=basename,
            expr=self.Phi0 / self.mesh_V0 * self.dV)



    def add_Js_qois(self):

        basename = "Js_"

        self.add_qoi(
            name=basename,
            expr=self.kinematics.Js / self.mesh_V0 * self.dV)



    def add_dWpordJs_qois(self):

        basename = "dWpordJs_"

        self.add_qoi(
            name=basename,
            expr=self.dWpordJs / self.mesh_V0 * self.dV)



    def add_dWbulkdJs_qois(self):

        basename = "dWbulkdJs_"

        self.add_qoi(
            name=basename,
            expr=self.dWbulkdJs / self.mesh_V0 * self.dV)



    def add_Phi0bin_qois(self):

        basename = "Phi0bin_"

        self.add_qoi(
            name=basename,
            expr=self.Phi0bin * self.dV)



    def add_psi_qois(self):

        basename = "Psi_"
        psi = 0
        for subdomain in self.subdomains :
            psi += subdomain.Psi * self.dV(subdomain.id)

        self.add_qoi(
            name=basename,
            expr=psi)



    def add_Wbulk_qois(self):

        basename = "Wbulk_"

        self.add_qoi(
            name=basename,
            expr=self.Wbulk * self.dV)
