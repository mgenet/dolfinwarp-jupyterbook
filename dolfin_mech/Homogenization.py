#coding=utf8

####################################################################################
###                                                                              ###
### Created by Martin Genet, Mahdi Manoocherhtayebi and Jeremy Bleyer 2018-2022  ###
###                                                                              ###
### École Polytechnique, Palaiseau, France                                       ###
###                                                                              ###
####################################################################################

import dolfin

import dolfin_mech as dmech

# from __future__ import print_function
import dolfin
import numpy as np

################################################################################

class HomogenizedParameters():



    def __init__(self,
            dim,
            mesh,
            E_s,
            nu_s):


        self.mesh = mesh
        self.E_s = E_s
        self.nu_s = nu_s
        self.dim = dim

        coord = self.mesh.coordinates()
        x_max = max(coord[:,0]); x_min = min(coord[:,0])
        y_max = max(coord[:,1]); y_min = min(coord[:,1])
        if (self.dim==3): z_max = max(coord[:,2]); z_min = min(coord[:,2])
        self.corners = np.array([[x_min, y_min],
                                 [x_max, y_min],
                                 [x_max, y_max],
                                 [x_min, y_max]])
        if (self.dim==2): 
            self.bbox = [x_min, x_max, y_min, y_max]
            self.vol = (x_max - x_min) * (y_max - y_min)
        elif (self.dim==3): 
            self.bbox = [x_min, x_max, y_min, y_max, z_min, z_max]
            self.vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        # print ("self.vol:", self.vol)


    
        self.material_parameters = [(self.E_s, self.nu_s)]


    def eps(self, v):
        return dolfin.sym(dolfin.grad(v))

    def sigma(self, v, i, Eps):
        E, nu = self.material_parameters[i]
        lmbda = E*nu/(1+nu)/(1-2*nu)
        mu = E/2/(1+nu)
        return lmbda*dolfin.tr(self.eps(v) + Eps) * dolfin.Identity(self.dim) + 2*mu*(self.eps(v)+Eps)

    def Voigt2strain(self, s):
        if (self.dim==2):
            strain_tensor = np.array([[s[0]   , s[2]/2.],
                                      [s[2]/2., s[1]   ]])
        if (self.dim==3):
            strain_tensor = np.array([[s[0]   , s[5]/2.,  s[4]/2],
                                      [s[5]/2., s[1]   ,  s[3]/2],
                                      [s[4]/2 , s[3]/2 ,  s[2]  ]])
        return strain_tensor

    def get_macro_strain(self, i):
        """returns the macroscopic strain for the 3 elementary load cases"""
        if (self.dim==2):
            Eps_Voigt = np.zeros(3)
        if (self.dim==3):
            Eps_Voigt = np.zeros(6)
        Eps_Voigt[i] = 1
        return self.Voigt2strain(Eps_Voigt)


    def stress2Voigt(self, s):
        if (self.dim==2):
            stress_vector = dolfin.as_vector([s[0,0], s[1,1], s[0,1]])
        if (self.dim==3):
            stress_vector = dolfin.as_vector([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]])
        return stress_vector

    def homogenized_param(self):
        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)
        Re = dolfin.VectorElement("R", self.mesh.ufl_cell(), 0)
        W = dolfin.FunctionSpace(self.mesh, dolfin.MixedElement([Ve, Re]), constrained_domain=dmech.PeriodicSubDomain(self.dim, self.bbox))
        V = dolfin.FunctionSpace(self.mesh, Ve)

        v_, lamb_ = dolfin.TestFunctions(W)
        dv, dlamb = dolfin.TrialFunctions(W)
        w = dolfin.Function(W)
        dx = dolfin.Measure('dx')(domain=self.mesh)

        if (self.dim==2): Eps = dolfin.Constant(((0, 0), (0, 0)))
        if (self.dim==3): Eps = dolfin.Constant(((0, 0, 0), (0, 0, 0), (0, 0, 0)))

        F = sum([dolfin.inner(self.sigma(dv, 0, Eps), self.eps(v_))*dx])

        a, L = dolfin.lhs(F), dolfin.rhs(F)
        a += dolfin.dot(lamb_,dv)*dx + dolfin.dot(dlamb,v_)*dx

        if (self.dim==2): Chom = np.zeros((3, 3))
        if (self.dim==3): Chom = np.zeros((6, 6))

        if (self.dim==2): Enum=enumerate(["Exx", "Eyy", "Exy"])
        if (self.dim==3): Enum=enumerate(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"])

        for (j, case) in Enum:
            print("Solving {} case...".format(case))
            macro_strain = self.get_macro_strain(j)
            Eps.assign(dolfin.Constant(macro_strain))
            dolfin.solve(a == L, w, [], solver_parameters={"linear_solver": "cg"})
            (v, lamb) = dolfin.split(w)
            # xdmf_file_per.write(w, float(j))

            if (self.dim==2): Sigma_len = 3
            if (self.dim==3): Sigma_len = 6

            Sigma = np.zeros((Sigma_len,))
            
            # self.vol = dolfin.assemble(dolfin.Constant(1) * dx)
            for k in range(Sigma_len):
                Sigma[k] = dolfin.assemble(self.stress2Voigt(self.sigma(v, 0, Eps))[k]*dx)/self.vol
            Chom[j, :] = Sigma
        
        # print(Chom)
        
        lmbda_hom = Chom[0, 1]
        if (self.dim==2): mu_hom = Chom[2, 2]
        if (self.dim==3): mu_hom = Chom[3, 3]

        # print(Chom[0, 0], lmbda_hom + 2*mu_hom)

        E_hom = mu_hom*(3*lmbda_hom + 2*mu_hom)/(lmbda_hom + mu_hom)
        nu_hom = lmbda_hom/(lmbda_hom + mu_hom)/2

        print("E_hom:" +str(E_hom))
        print("nu_hom:" +str(nu_hom))

        return lmbda_hom, mu_hom

