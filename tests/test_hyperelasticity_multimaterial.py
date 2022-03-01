#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

#################################################################### imports ###

import dolfin
import sys

import myPythonLibrary as mypy
import dolfin_mech     as dmech

################################################################################

def test_hyperelasticity_multimaterial(
    dim,
    incomp,
    load, # disp, volu, surf, pres
    res_basename,
    verbose=0):

    ################################################################### Mesh ###

    LX = 1.
    LY = 1.
    if (dim==3): LZ = 1.

    l = 0.1
    NX = int(LX/l)
    NY = int(LY/l)
    if (dim==3): NZ = int(LZ/l)

    if (dim==2):
        mesh = dolfin.RectangleMesh(
            dolfin.Point(0., 0., 0.),
            dolfin.Point(LX, LY, 0.),
            NX,
            NY,
            "crossed")
    elif (dim==3):
        mesh = dolfin.BoxMesh(
            dolfin.Point(0., 0., 0.),
            dolfin.Point(LX, LY, LZ),
            NX,
            NY,
            NZ)

    # xdmf_file_mesh = dolfin.XDMFFile(res_basename+"-mesh.xdmf")
    # xdmf_file_mesh.write(mesh)
    # xdmf_file_mesh.close()

    ################################################## Subdomains & Measures ###

    xmin_sd = dolfin.CompiledSubDomain("near(x[0], x0) && on_boundary", x0=0.)
    xmax_sd = dolfin.CompiledSubDomain("near(x[0], x0) && on_boundary", x0=LX)
    ymin_sd = dolfin.CompiledSubDomain("near(x[1], x0) && on_boundary", x0=0.)
    ymax_sd = dolfin.CompiledSubDomain("near(x[1], x0) && on_boundary", x0=LY)
    if (dim==3): zmin_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=0.)
    if (dim==3): zmax_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=LZ)

    xmin_id = 1
    xmax_id = 2
    ymin_id = 3
    ymax_id = 4
    if (dim==3): zmin_id = 5
    if (dim==3): zmax_id = 6

    boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # MG20180418: size_t looks like unisgned int, but more robust wrt architecture and os
    boundaries_mf.set_all(0)
    xmin_sd.mark(boundaries_mf, xmin_id)
    xmax_sd.mark(boundaries_mf, xmax_id)
    ymin_sd.mark(boundaries_mf, ymin_id)
    ymax_sd.mark(boundaries_mf, ymax_id)
    if (dim==3): zmin_sd.mark(boundaries_mf, zmin_id)
    if (dim==3): zmax_sd.mark(boundaries_mf, zmax_id)

    # xdmf_file_boundaries = dolfin.XDMFFile(res_basename+"-boundaries.xdmf")
    # xdmf_file_boundaries.write(boundaries_mf)
    # xdmf_file_boundaries.close()

    mat1_sd = dolfin.CompiledSubDomain("x[0] <= x0", x0=LX/2)
    mat2_sd = dolfin.CompiledSubDomain("x[0] >= x0", x0=LX/2)

    mat1_id = 1
    mat2_id = 2

    domains_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()) # MG20180418: size_t looks like unisgned int, but more robust wrt architecture and os
    domains_mf.set_all(0)
    mat1_sd.mark(domains_mf, mat1_id)
    mat2_sd.mark(domains_mf, mat2_id)

    # xdmf_file_domains = dolfin.XDMFFile(res_basename+"-domains.xdmf")
    # xdmf_file_domains.write(domains_mf)
    # xdmf_file_domains.close()

    ###################################################### Material behavior ###

    mat1_params = {
        "E":1.,
        "nu":0.5*(incomp)+0.3*(1-incomp)}

    mat2_params = {
        "E":10.,
        "nu":0.5*(incomp)+0.3*(1-incomp)}

    if (incomp):
        mat1 = dmech.NeoHookeanMooneyRivlinDevElasticMaterial(
            parameters=mat1_params)
        mat2 = dmech.NeoHookeanMooneyRivlinDevElasticMaterial(
            parameters=mat2_params)
    else:
        mat1 = dmech.CiarletGeymonatNeoHookeanMooneyRivlinElasticMaterial(
            parameters=mat1_params)
        mat2 = dmech.CiarletGeymonatNeoHookeanMooneyRivlinElasticMaterial(
            parameters=mat2_params)

    ################################################################ Problem ###

    if (incomp):
        problem = dmech.HyperelasticityProblem(
            mesh=mesh,
            domains_mf=domains_mf,
            compute_normals=1,
            boundaries_mf=boundaries_mf,
            U_degree=2, # MG20211219: Incompressibility requires U_degree >= 2 ?!
            quadrature_degree="default",
            w_incompressibility=1,
            elastic_behavior_dev=[(mat1_id, mat1), (mat2_id, mat2)])
    else:
        problem = dmech.HyperelasticityProblem(
            mesh=mesh,
            domains_mf=domains_mf,
            compute_normals=1,
            boundaries_mf=boundaries_mf,
            U_degree=1,
            quadrature_degree="default",
            w_incompressibility=0,
            elastic_behavior=[(mat1_id, mat1), (mat2_id, mat2)])

    ########################################## Boundary conditions & Loading ###

    # problem.add_constraint(V=problem.get_displacement_function_space(), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=[0.]*dim)
    # problem.add_constraint(V=problem.get_displacement_function_space(), sub_domains=boundaries_mf, sub_domain_id=xmax_id, val=[0.]*dim)
    # problem.add_constraint(V=problem.get_displacement_function_space(), sub_domains=boundaries_mf, sub_domain_id=ymin_id, val=[0.]*dim)
    # problem.add_constraint(V=problem.get_displacement_function_space(), sub_domains=boundaries_mf, sub_domain_id=ymax_id, val=[0.]*dim)
    # if (dim==3): problem.add_constraint(V=problem.get_displacement_function_space(), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=[0.]*dim)
    # if (dim==3): problem.add_constraint(V=problem.get_displacement_function_space(), sub_domains=boundaries_mf, sub_domain_id=zmax_id, val=[0.]*dim)

    problem.add_constraint(V=problem.get_displacement_function_space().sub(0), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=0.)
    # problem.add_constraint(V=problem.get_displacement_function_space().sub(0), sub_domains=boundaries_mf, sub_domain_id=xmax_id, val=0.)
    problem.add_constraint(V=problem.get_displacement_function_space().sub(1), sub_domains=boundaries_mf, sub_domain_id=ymin_id, val=0.)
    # problem.add_constraint(V=problem.get_displacement_function_space().sub(1), sub_domains=boundaries_mf, sub_domain_id=ymax_id, val=0.)
    if (dim==3): problem.add_constraint(V=problem.get_displacement_function_space().sub(2), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.)
    # if (dim==3): problem.add_constraint(V=problem.get_displacement_function_space().sub(2), sub_domains=boundaries_mf, sub_domain_id=zmax_id, val=0.)

    k_step = problem.add_step(
        Deltat=1.,
        dt_ini=1.,
        dt_min=0.1)

    if (load == "disp"):
        problem.add_constraint(
            V=problem.get_displacement_function_space().sub(0),
            sub_domains=boundaries_mf,
            sub_domain_id=xmax_id,
            val_ini=0.0,
            val_fin=0.5,
            k_step=k_step)
    elif (load == "volu0"):
        problem.add_volume_force0_loading_operator(
            measure=problem.dV,
            F_ini=[0.0]*dim,
            F_fin=[0.5]+[0.0]*(dim-1),
            k_step=k_step)
    elif (load == "volu"):
        problem.add_volume_force_loading_operator(
            measure=problem.dV,
            F_ini=[0.0]*dim,
            F_fin=[1.0]+[0.0]*(dim-1),
            k_step=k_step)
    elif (load == "surf0"):
        problem.add_surface_force0_loading_operator(
            measure=problem.dS(xmax_id),
            F_ini=[0.0]*dim,
            F_fin=[1.0]+[0.0]*(dim-1),
            k_step=k_step)
    elif (load == "surf"):
        problem.add_surface_force_loading_operator(
            measure=problem.dS(xmax_id),
            F_ini=[0.0]*dim,
            F_fin=[1.0]+[0.0]*(dim-1),
            k_step=k_step)
    elif (load == "pres0"):
        problem.add_pressure0_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=-0.0,
            P_fin=-0.5,
            k_step=k_step)
    elif (load == "pres"):
        problem.add_pressure_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=-0.0,
            P_fin=-0.5,
            k_step=k_step)
    elif (load == "pgra0"):
        problem.add_pressure_gradient0_loading_operator(
            measure=problem.dS(),
            X0=[0.5]*dim,
            N0=[1.]+[0.]*(dim-1),
            P0_ini=-0.0,
            P0_fin=-0.5,
            DP_ini=-0.00,
            DP_fin=-0.25,
            k_step=k_step)
    elif (load == "pgra"):
        problem.add_pressure_gradient_loading_operator(
            measure=problem.dS(),
            X0=[0.5]*dim,
            N0=[1.]+[0.]*(dim-1),
            P0_ini=-0.0,
            P0_fin=-0.5,
            DP_ini=-0.00,
            DP_fin=-0.25,
            k_step=k_step)
    elif (load == "tens"):
        problem.add_surface_tension_loading_operator(
            measure=problem.dS,
            gamma_ini=0.00,
            gamma_fin=0.01,
            k_step=k_step)

    ################################################# Quantities of Interest ###

    problem.add_global_strain_qois()
    problem.add_global_stress_qois()
    if (incomp): problem.add_global_pressure_qoi()

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

################################################################################

if (__name__ == "__main__"):

    res_folder = sys.argv[0][:-3]
    test = mypy.Test(
        res_folder=res_folder,
        perform_tests=1,
        stop_at_failure=1,
        clean_after_tests=1)

    dim_lst  = []
    dim_lst += [2]
    dim_lst += [3]
    for dim in dim_lst:

        incomp_lst  = []
        incomp_lst += [0]
        incomp_lst += [1]
        for incomp in incomp_lst:

            print("dim =",dim)
            print("incomp =",incomp)

            res_basename  = sys.argv[0][:-3]
            res_basename += "-dim="+str(dim)
            res_basename += "-incomp="+str(incomp)

            test_hyperelasticity_multimaterial(
                dim=dim,
                incomp=incomp,
                load="disp",
                res_basename=res_folder+"/"+res_basename,
                verbose=0)

            test.test(res_basename)