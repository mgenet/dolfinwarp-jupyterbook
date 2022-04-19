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

# from __future__ import print_function
import dolfin
import numpy as np

################################################################################

class HomogenizedParameters():



    def __init__(self,
            base_name,
            E_s,
            nu_s):

        self.base_name = base_name
        self.E_s = E_s
        self.nu_s = nu_s

        self.mesh = dolfin.Mesh(self.base_name + ".xml")
        self.subdomains = dolfin.MeshFunction("size_t", self.mesh, self.base_name + "_physical_region.xml")
        self.facets = dolfin.MeshFunction("size_t", self.mesh, self.base_name + "_facet_region.xml") 

        coord = self.mesh.coordinates()
        x_max = max(coord[:,0]); x_min = min(coord[:,0])
        y_max = max(coord[:,1]); y_min = min(coord[:,1])
        self.corners = np.array([[x_min, y_min],
                                 [x_max, y_min],
                                 [x_max, y_max],
                                 [x_min, y_max]])

        self.vol = (x_max - x_min) * (y_max - y_min)

      

        self.material_parameters = [(self.E_s, self.nu_s)]


    def eps(self, v):
        return dolfin.sym(dolfin.grad(v))

    def sigma(self, v, i, Eps):
        E, nu = self.material_parameters[i]
        lmbda = E*nu/(1+nu)/(1-2*nu)
        mu = E/2/(1+nu)
        return lmbda*dolfin.tr(self.eps(v) + Eps) * dolfin.Identity(2) + 2*mu*(self.eps(v)+Eps)

    def Voigt2strain(self, s):
         return np.array([[s[0]   , s[2]/2.],
                          [s[2]/2., s[1]   ]])

    def get_macro_strain(self, i):
         """returns the macroscopic strain for the 3 elementary load cases"""
         Eps_Voigt = np.zeros(3)
         Eps_Voigt[i] = 1
         return self.Voigt2strain(Eps_Voigt)


    def stress2Voigt(self, s):
        return dolfin.as_vector([s[0,0], s[1,1], s[0,1]])


    def homogenized_param(self):
        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)
        Re = dolfin.VectorElement("R", self.mesh.ufl_cell(), 0)
        W = dolfin.FunctionSpace(self.mesh, dolfin.MixedElement([Ve, Re]), constrained_domain=dmech.Periodic_Boundary(self.corners, tolerance=1e-7))
        V = dolfin.FunctionSpace(self.mesh, Ve)

        v_,lamb_ = dolfin.TestFunctions(W)
        dv, dlamb = dolfin.TrialFunctions(W)
        w = dolfin.Function(W)
        dx = dolfin.Measure('dx')(domain=self.mesh)

        Eps = dolfin.Constant(((0, 0), (0, 0)))

        F = sum([dolfin.inner(self.sigma(dv, 0, Eps), self.eps(v_))*dx])

        a, L = dolfin.lhs(F), dolfin.rhs(F)
        a += dolfin.dot(lamb_,dv)*dx + dolfin.dot(dlamb,v_)*dx

        Chom = np.zeros((3, 3))

        for (j, case) in enumerate(["Exx", "Eyy", "Exy"]):
            print("Solving {} case...".format(case))
            macro_strain = self.get_macro_strain(j)
            Eps.assign(dolfin.Constant(macro_strain))
            dolfin.solve(a == L, w, [], solver_parameters={"linear_solver": "cg"})
            (v, lamb) = dolfin.split(w)
            # xdmf_file_per.write(w, float(j))
            Sigma = np.zeros((3,))
            for k in range(3):
                Sigma[k] = dolfin.assemble(self.stress2Voigt(self.sigma(v, 0, Eps))[k]*dx)/self.vol
            Chom[j, :] = Sigma

        lmbda_hom = Chom[0, 1]
        mu_hom = Chom[2, 2]

        E_hom = mu_hom*(3*lmbda_hom + 2*mu_hom)/(lmbda_hom + mu_hom)
        nu_hom = lmbda_hom/(lmbda_hom + mu_hom)/2

        return E_hom, nu_hom

