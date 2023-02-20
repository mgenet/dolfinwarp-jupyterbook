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
import math
import sys
################################################################################

class HomogenizedParameters():



    def __init__(self,
            dim,
            mesh,
            mat_params,
            vertices,
            vol,
            bbox):


        self.mesh = mesh
        self.E_s = mat_params["E"]
        self.nu_s = mat_params["nu"]
        self.dim = dim
        self.vertices=vertices
        self.vol=vol

        self.bbox=bbox
        print ("self.vol:", self.vol)


    
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
        W = dolfin.FunctionSpace(self.mesh, dolfin.MixedElement([Ve, Re]), constrained_domain=dmech.PeriodicSubDomain(self.dim, self.bbox, self.vertices))
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

        print("E_hom: " +str(E_hom))
        print("nu_hom: " +str(nu_hom))
        # print(Chom[0, 0], lmbda_hom + 2*mu_hom)
        print("Isotropy = " +str ((lmbda_hom + 2*mu_hom - abs(lmbda_hom + 2*mu_hom - Chom[0, 0]))/(lmbda_hom + 2*mu_hom) * 100) +"%")

        self.E_hom = E_hom
        self.nu_hom = nu_hom 
        return lmbda_hom, mu_hom


    def kappa_tilde(self):
        p_f = 0.001

  
        load = "internal_pressure"

        dim = 2
        mat_params = {"model":"CGNH", "parameters":{"E":self.E_s, "nu":self.nu_s}}
        bcs = "pbc"
        step_params = {"dt_ini":1e-1, "dt_min":1e-3}
        load_params = {"type":load}

        res_folder = sys.argv[0][:-3]
        res_basename  = sys.argv[0][:-3]
        res_basename += "-dim="+str(dim)
        res_basename += "-bcs="+str(bcs)
        res_basename += "-load="+str(load)

        res_basename = res_folder+"/"+res_basename
        verbose=0

        ################################################################### Mesh ###

        xmin = self.bbox[0]
        xmax = self.bbox[1]
        ymin = self.bbox[2]
        ymax = self.bbox[3]

        tol = 1E-8
        vv = self.vertices
        a1 = vv[1,:]-vv[0,:] # first vector generating periodicity
        a2 = vv[3,:]-vv[0,:] # second vector generating periodicity
        # check if UC vertices form indeed a parallelogram
        assert np.linalg.norm(vv[2, :]-vv[3, :] - a1) <= tol

        ################################################## Subdomains & Measures ###

        class BoundaryX0(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                # return on_boundary and dolfin.near(x[0],xmin,tol)
                return on_boundary and dolfin.near(x[0],vv[0,0] + x[1]*a2[0]/vv[3,1],tol)

        class BoundaryY0(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and dolfin.near(x[1],ymin,tol)

        class BoundaryX1(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                # return on_boundary and dolfin.near(x[0],xmax,tol)
                return on_boundary and dolfin.near(x[0],vv[1,0] + x[1]*a2[0]/vv[3,1],tol)

        class BoundaryY1(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and dolfin.near(x[1],ymax,tol)

        boundaries_mf = dolfin.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        boundaries_mf.set_all(0)

        # bsup = BoundaryHigher()
        bX0 = BoundaryX0()
        bY0 = BoundaryY0()
        bX1 = BoundaryX1()
        bY1 = BoundaryY1()

        # bsup.mark(boundary_markers, 8)
        bX0.mark(boundaries_mf, 1)
        bY0.mark(boundaries_mf, 2)
        bX1.mark(boundaries_mf, 3)
        bY1.mark(boundaries_mf, 4)

        ################################################################ Problem ###

        problem = dmech.MicroPoroElasticityProblem(
                pf=p_f,
                mesh=self.mesh,
                mesh_bbox=self.bbox,
                bbox_V0=self.vol,
                vertices=self.vertices,
                boundaries_mf=boundaries_mf,
                displacement_perturbation_degree=2,
                quadrature_degree=3,
                solid_behavior=mat_params,
                bcs=bcs)

        ################################################################ Loading ###

        Deltat = step_params.get("Deltat", 1.)
        dt_ini = step_params.get("dt_ini", 1.)
        dt_min = step_params.get("dt_min", 1.)
        dt_max = step_params.get("dt_max", 1.)
        k_step = problem.add_step(
            Deltat=Deltat,
            dt_ini=dt_ini,
            dt_min=dt_min,
            dt_max=dt_max)

        sigma_bar_00 = None
        sigma_bar_01 = None
        sigma_bar_10 = None
        sigma_bar_11 = None
        sigma_bar = [[sigma_bar_00, sigma_bar_01],
                    [sigma_bar_10, sigma_bar_11]]
        load_type = load_params.get("type", "internal_pressure")
        # if (load_type == "internal_pressure"):
        pf = load_params.get("pf", p_f)
        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(0),
            P_ini=0., P_fin=pf,
            k_step=k_step)
        # # elif (load_type == "macroscopic_stretch"):
        problem.add_macroscopic_stretch_component_penalty_operator(
            comp_i=0, comp_j=0,
            comp_ini=0.0, comp_fin=0,
            pen_val=1e6,
            k_step=k_step)
        problem.add_macroscopic_stretch_component_penalty_operator(
            comp_i=1, comp_j=1,
            comp_ini=0.0, comp_fin=0,
            pen_val=1e6,
            k_step=k_step)
        problem.add_macroscopic_stretch_component_penalty_operator(
            comp_i=0, comp_j=1,
            comp_ini=0.0, comp_fin=0.0,
            pen_val=1e6,
            k_step=k_step)
        # elif (load_type == "macroscopic_stress"):
        # gamma = load_params.get("gamma", 0.004)
        # problem.add_surface_tension_loading_operator_1(
        #     measure=problem.dS,
        #     gamma_ini=0, gamma_fin=gamma,
        #     k_step=k_step)

        for k in range(dim):
            for l in range (dim):
                if sigma_bar[k][l] is None:
                    problem.add_macroscopic_stress_lagrange_multiplier_component_penalty_operator(
                        i=k, j=l,
                        pen_val=1e6,
                        k_step=k_step)
                else:
                    problem.add_macroscopic_stress_component_constraint_operator(
                        i=k, j=l,
                        sigma_bar_ij_ini=0.0, sigma_bar_ij_fin=sigma_bar[k][l],
                        pf_ini=0.0, pf_fin=pf,
                        k_step=k_step)

        ################################################################# Solver ###

        solver = dmech.NonlinearSolver(
            problem=problem,
            parameters={
                "sol_tol":[1e-6]*len(problem.subsols),
                "n_iter_max":32},
            relax_type="constant",
            write_iter=0)

        integrator = dmech.TimeIntegrator(
            problem=problem,
            solver=solver,
            parameters={
                "n_iter_for_accel":4,
                "n_iter_for_decel":16,
                "accel_coeff":2,
                "decel_coeff":2},
            print_out=res_basename*verbose,
            print_sta=res_basename*verbose,
            write_qois=res_basename+"-qois",
            write_qois_limited_precision=1,
            write_sol=res_basename*verbose)

        success = integrator.integrate()
        assert (success),\
            "Integration failed. Aborting."

        integrator.close()
        Vs0 = problem.mesh_V0
        vs = dolfin.assemble(problem.kinematics.J * problem.dV)
        V0 = problem.bbox_V0
        print("vs: "+str(vs))
        print("Vs0: "+str(Vs0))
        Phi_s0 = Vs0/V0
        Phi_s = vs/V0

        kappa_tilde = Phi_s0**2/(Phi_s0 - Phi_s)/4 * p_f

        print("kappa_tilde: " +str(kappa_tilde))

        return kappa_tilde