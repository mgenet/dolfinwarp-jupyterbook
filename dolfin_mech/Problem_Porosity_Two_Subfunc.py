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
# import math

import dolfin_mech as dmech
from .Problem_Hyperelasticity import HyperelasticityProblem

################################################################################

class TwoSubfuncPoroProblem(HyperelasticityProblem):



    def __init__(self,
            eta,
            kappa,
            w_contact = 1,
            p0 = 0):

        HyperelasticityProblem.__init__(self,w_incompressibility=False)
        self.eta                 = eta
        self.kappa               = kappa
        self.p0                  = p0
        self.w_contact           = w_contact
        self.inertia             = None
        self.porosity_init_val   = None
        self.porosity_init_field = None
        self.porosity_given      = None
        self.config_porosity     = None

        assert (w_contact == 0) or (eta == 0)



    def add_porosity_subsol(self,
            degree):

        if self.porosity_init_val is not None:
            init_val = numpy.array([self.porosity_init_val])
        else:
            init_val = self.porosity_init_val

        if (degree == 0):
            self.add_scalar_subsol(
                name="Phi",
                family="DG",
                degree=0,
                init_val=init_val,
                init_field=self.porosity_init_field)
        else:
            self.add_scalar_subsol(
                name="Phi",
                family="CG",
                degree=degree,
                init_val=init_val,
                init_field=self.porosity_init_field)



    def set_subsols(self,
            U_degree=1):

        self.add_displacement_subsol(
            degree=U_degree)

        self.add_porosity_subsol(
            degree=U_degree-1)



    def get_porosity_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="Phi")



    def set_porosity_energy(self):

        # exp
        dWpordJs = self.eta * (self.Phi0 / (self.kinematics.Je * self.Phi))**2 * dolfin.exp(-self.kinematics.Je * self.Phi / (self.Phi0 - self.kinematics.Je * self.Phi)) / (self.Phi0 - self.kinematics.Je * self.Phi)
        # n = 2
        # dWpordJs = self.eta * n * ((self.Phi0 - self.kinematics.Je * self.Phi) / (self.kinematics.Je * self.Phi))**(n-1) * self.Phi0 / (self.kinematics.Je * self.Phi)**2
        # dWpordJs = 0
        dWpordJs_condition = dolfin.conditional(dolfin.lt(self.Phi, self.Phi0), dWpordJs, 0)
        self.dWpordJs = (1 - self.Phi0) * dWpordJs_condition



    def set_bulk_energy(self):

        dWbulkdJs = self.kappa * (1. / (1. - self.Phi0) - 1./self.kinematics.Js)
        self.dWbulkdJs = (1 - self.Phi0) * dWbulkdJs

        if self.w_contact:
            dWbulkdJs_pos = self.kappa * (1. / (1. - self.Phi0pos) - 1./self.kinematics.Js_pos)
            self.dWbulkdJs_pos = (1 - self.Phi0pos) * dWbulkdJs_pos
        else:
            self.dWbulkdJs_pos = self.dWbulkdJs



    def set_Phi0_and_Phi(self):

        if self.config_porosity == 'ref':
            self.Phi0 = self.porosity_given
            self.Phi0pos = dolfin.conditional(dolfin.gt(self.Phi0,0), self.Phi0, 0)
            self.Phi0bin = dolfin.conditional(dolfin.gt(self.Phi0,0), 1, 0)
            self.Phi  = self.subsols["Phi"].subfunc
            self.Phipos  = dolfin.conditional(dolfin.gt(self.Phi,0), self.Phi, 0)
            self.Phibin = dolfin.conditional(dolfin.gt(self.Phi,0), 1, 0)
        elif self.config_porosity == 'deformed':
            self.Phi0 = Nan
            self.Phi  = Nan



    def set_kinematics(self):

        HyperelasticityProblem.set_kinematics(self)

        self.set_Phi0_and_Phi()
        self.kinematics.Js = self.kinematics.Je * (1 - self.Phi)
        self.kinematics.Js_pos = self.kinematics.Je * (1 - self.Phipos)



    def set_materials(self,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None,
            subdomain_id=None):

        self.set_kinematics()

        HyperelasticityProblem.set_materials(self,
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

        self.Pi = sum([subdomain.Psi * self.dV(subdomain.id) for subdomain in self.subdomains])
        # print (self.Pi)

        self.res_form = dolfin.derivative(
            self.Pi,
            self.sol_func,
            self.dsol_test);

        if self.inertia is not None:
            self.res_form += self.inertia / dt * dolfin.inner(
                    self.subsols["U"].subfunc,
                    self.subsols["U"].dsubtest) * self.dV

        for loading in pressure_loadings:
            T = dolfin.dot(
               -loading.val * self.mesh_normals,
                dolfin.inv(self.kinematics.Ft))
            self.res_form -= self.kinematics.Jt * dolfin.inner(
                T,
                self.subsols["U"].dsubtest) * loading.measure

        self.res_form += dolfin.inner(
            self.dWbulkdJs_pos * self.kinematics.Je * self.kinematics.Ce_inv,
            dolfin.derivative(
                    self.kinematics.Et,
                    self.subsols["U"].subfunc,
                    self.subsols["U"].dsubtest)) * self.dV

        p0_loading_val = pressure0_loadings[0].val
        self.res_form += dolfin.inner(
                self.dWbulkdJs + self.dWpordJs + p0_loading_val,
                self.subsols["Phi"].dsubtest) * self.dV

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)



    def add_Phi0_qois(self):

        basename = "PHI0_"

        self.add_qoi(
            name=basename,
            expr=self.Phi0 / self.mesh_V0 * self.dV)



    def add_Phi_qois(self):

        basename = "PHI_"

        self.add_qoi(
            name=basename,
            expr=self.Phi / self.mesh_V0 * self.dV)



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

        basename = "PHI0bin_"

        self.add_qoi(
            name=basename,
            expr=self.Phi0bin / self.mesh_V0 * self.dV)



    def add_Phi0pos_qois(self):

        basename = "PHI0pos_"

        self.add_qoi(
            name=basename,
            expr=self.Phi0pos / self.mesh_V0 * self.dV)



    def add_mnorm_qois(self):

        basename = "M_NORM"
        value = self.kinematics.Je * self.Phi - self.Phi0

        self.add_qoi(
            name=basename,
            expr=value / self.mesh_V0 * self.dV)



    def add_Phibin_qois(self):

        basename = "PHIbin_"

        self.add_qoi(
            name=basename,
            expr=self.Phibin / self.mesh_V0 * self.dV)



    def add_Phipos_qois(self):

        basename = "PHIpos_"

        self.add_qoi(
            name=basename,
            expr=self.Phipos / self.mesh_V0 * self.dV)
