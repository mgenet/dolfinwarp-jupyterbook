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

################################################################################

class Material():



    def get_lambda_from_parameters(self,
            parameters,
            dim=3,
            PS=False):

        if ("lambda" in parameters):
            lmbda = parameters["lambda"]
        elif ("E" in parameters) and ("nu" in parameters):
            E  = parameters["E"]
            nu = parameters["nu"]
            if (dim == 2) and (PS):
                lmbda = E*nu/(1+nu)/(1-  nu)
            else:
                lmbda = E*nu/(1+nu)/(1-2*nu)
        else:
            assert (0),\
                "No parameter found: \"+str(parameters)+\". Must provide lambda or E & nu. Aborting."
        return dolfin.Constant(lmbda)



    def get_mu_from_parameters(self,
            parameters):

        if ("mu" in parameters):
            mu = parameters["mu"]
        elif ("E" in parameters) and ("nu" in parameters):
            E  = parameters["E"]
            nu = parameters["nu"]
            mu = E/2/(1+nu)
        else:
            assert (0),\
                "No parameter found: \"+str(parameters)+\". Must provide mu or E & nu. Aborting."
        return dolfin.Constant(mu)



    def get_K_from_parameters(self,
            parameters,
            dim=3,
            PS=False):

        if ("K" in parameters):
            K = parameters["K"]
        else:
            if ("lambda" in parameters) and ("mu" in parameters):
                lmbda = parameters["lambda"]
                mu    = parameters["mu"]
            elif ("E" in parameters) and ("nu" in parameters):
                E  = parameters["E"]
                nu = parameters["nu"]
                if (dim == 2) and (PS):
                    lmbda = E*nu/(1+nu)/(1-  nu)
                else:
                    lmbda = E*nu/(1+nu)/(1-2*nu)
                mu = E/2/(1+nu)
            else:
                assert (0),\
                    "No parameter found: \"+str(parameters)+\". Must provide K or lambda & mu or E & nu. Aborting."
            if   (dim == 2):
                K = (2*lmbda+2*mu)/2
            elif (dim == 3):
                K = (3*lmbda+2*mu)/3
            else:
                assert (0),\
                    "Only in 2D & 3D. Aborting."
        return dolfin.Constant(K)



    def get_G_from_parameters(self,
            parameters,
            dim=3,
            PS=False):

        if ("G" in parameters):
            G = parameters["G"]
        else:
            if ("mu" in parameters):
                mu = parameters["mu"]
            elif ("E" in parameters) and ("nu" in parameters):
                E  = parameters["E"]
                nu = parameters["nu"]
                mu = E/2/(1+nu)
            else:
                assert (0),\
                    "No parameter found: \"+str(parameters)+\". Must provide G or mu or E & nu. Aborting."
            G = mu
        return dolfin.Constant(G)


    
    def get_C1_from_parameters(self,
            parameters):

        if ("C1" in parameters):
            C1 = parameters["C1"]
        elif ("c1" in parameters):
            C1 = parameters["c1"]
        elif ("mu" in parameters):
            mu = parameters["mu"]
            C1 = mu/2
        elif ("E" in parameters) and ("nu" in parameters):
            E  = parameters["E"]
            nu = parameters["nu"]
            mu = E/2/(1+nu)
            C1 = mu/2
        else:
            assert (0),\
                "No parameter found ("+str(parameters)+"). Must provide C1 or c1 or mu or E & nu. Aborting."
        return dolfin.Constant(C1)



    def get_C2_from_parameters(self,
            parameters):

        if ("C2" in parameters):
            C2 = parameters["C2"]
        elif ("c2" in parameters):
            C2 = parameters["c2"]
        elif ("mu" in parameters):
            mu = parameters["mu"]
            C2 = mu/2
        elif ("E" in parameters) and ("nu" in parameters):
            E  = parameters["E"]
            nu = parameters["nu"]
            mu = E/2/(1+nu)
            C2 = mu/2
        else:
            assert (0),\
                "No parameter found ("+str(parameters)+"). Must provide C2 or c2 or mu or E & nu. Aborting."
        return dolfin.Constant(C2)



    def get_C1_and_C2_from_parameters(self,
            parameters):

        if ("C1" in parameters) and ("C2" in parameters):
            C1 = parameters["C1"]
            C2 = parameters["C2"]
        elif ("c1" in parameters) and ("c2" in parameters):
            C1 = parameters["c1"]
            C2 = parameters["c2"]
        elif ("mu" in parameters):
            mu = parameters["mu"]
            C1 = mu/4
            C2 = mu/4
        elif ("E" in parameters) and ("nu" in parameters):
            E  = parameters["E"]
            nu = parameters["nu"]
            mu = E/2/(1+nu)
            C1 = mu/4
            C2 = mu/4
        else:
            assert (0),\
                "No parameter found ("+str(parameters)+"). Must provide C1 & C2 or c1 & c2 or mu or E & nu. Aborting."
        return dolfin.Constant(C1), dolfin.Constant(C2)


    
    def get_C_from_U_or_C(self,
            U=None,
            C=None):

        if (C is not None):
            assert (C.ufl_shape[0] == C.ufl_shape[1])
        elif (U is not None):
            dim = U.ufl_shape[0]
            I = dolfin.Identity(dim)
            F = I + dolfin.grad(U)
            # JF = dolfin.det(F) # MG20211220: Otherwise cannot derive Psi wrt C
            C = F.T * F
        else:
            assert (0),\
                "Must provide U or C. Aborting."
        return dolfin.variable(C)



    def get_E_from_U_C_or_E(self,
            U=None,
            C=None,
            E=None):

        if (E is not None):
            assert (E.ufl_shape[0] == E.ufl_shape[1])
        elif (U is not None) or (C is not None):
            C = self.get_C_from_U_or_C(U, C)
            dim = C.ufl_shape[0]
            I = dolfin.Identity(dim)
            E = (C - I)/2
        else:
            assert (0),\
                "Must provide U, C or E. Aborting."
        return dolfin.variable(E)



    def get_E_sph_from_U_C_E_or_E_sph(self,
            U=None,
            C=None,
            E=None,
            E_sph=None):

        if (E_sph is not None):
            assert (E_sph.ufl_shape[0] == E_sph.ufl_shape[1])
        elif (U is not None) or (C is not None) or (E is not None):
            E = self.get_E_from_U_C_or_E(U, C, E)
            dim = E.ufl_shape[0]
            I = dolfin.Identity(dim)
            E_sph = dolfin.tr(E)/dim * I
        else:
            assert (0),\
                "Must provide U, C, E or E_sph. Aborting."
        return dolfin.variable(E_sph)



    def get_E_dev_from_U_C_E_or_E_dev(self,
            U=None,
            C=None,
            E=None,
            E_dev=None):

        if (E_dev is not None):
            assert (E_dev.ufl_shape[0] == E_dev.ufl_shape[1])
        elif (U is not None) or (C is not None) or (E is not None):
            E = self.get_E_from_U_C_or_E(U, C, E)
            E_sph = self.get_E_sph_from_U_C_E_or_E_sph(U, C, E)
            E_dev = E - E_sph
        else:
            assert (0),\
                "Must provide U, C, E or E_dev. Aborting."
        return dolfin.variable(E_dev)



    def get_epsilon_from_U_or_epsilon(self,
            U=None,
            epsilon=None):

        if (epsilon is not None):
            assert (epsilon.ufl_shape[0] == epsilon.ufl_shape[1])
        elif (U is not None):
            epsilon = dolfin.sym(dolfin.grad(U))
        else:
            assert (0),\
                "Must provide U or epsilon. Aborting."
        return dolfin.variable(epsilon)



    def get_epsilon_sph_from_U_epsilon_or_epsilon_sph(self,
            U=None,
            epsilon=None,
            epsilon_sph=None):

        if (epsilon_sph is not None):
            assert (epsilon_sph.ufl_shape[0] == epsilon_sph.ufl_shape[1])
        elif (U is not None) or (epsilon is not None):
            epsilon = self.get_epsilon_from_U_or_epsilon(U, epsilon)
            dim = epsilon.ufl_shape[0]
            I = dolfin.Identity(dim)
            epsilon_sph = dolfin.tr(epsilon)/dim * I
        else:
            assert (0),\
                "Must provide U, epsilon or epsilon_sph. Aborting."
        return dolfin.variable(epsilon_sph)



    def get_epsilon_dev_from_U_epsilon_or_epsilon_dev(self,
            U=None,
            epsilon=None,
            epsilon_dev=None):

        if (epsilon_dev is not None):
            assert (epsilon_dev.ufl_shape[0] == epsilon_dev.ufl_shape[1])
        elif (U is not None) or (epsilon is not None):
            epsilon = self.get_epsilon_from_U_or_epsilon(U, epsilon)
            epsilon_sph = self.get_epsilon_sph_from_U_epsilon_or_epsilon_sph(U, epsilon)
            epsilon_dev = epsilon - epsilon_sph
        else:
            assert (0),\
                "Must provide U, epsilon or epsilon_dev. Aborting."
        return dolfin.variable(epsilon_dev)
