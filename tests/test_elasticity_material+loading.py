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

def test_elasticity(
    dim,
    PS,
    incomp,
    load, # disp, volu, surf, pres, pgra, tens
    res_basename,
    verbose=0):

    ################################################################### Mesh ###

    if   (dim==2):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.init_Rivlin_cube(dim=dim)
    elif (dim==3):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = dmech.init_Rivlin_cube(dim=dim)

    ################################################################ Problem ###

    quadrature_degree = "default"
    # quadrature_degree = "full"

    material_parameters = {
        "E":1.,
        "nu":0.5*(incomp)+0.3*(1-incomp),
        "dim":dim, # MG20220322: Necessary to compute correct bulk modulus
        "PS":PS}

    if (incomp):
        problem = dmech.ElasticityProblem(
            mesh=mesh,
            compute_normals=1,
            boundaries_mf=boundaries_mf,
            u_degree=2, # MG20211219: Incompressibility requires U_degree >= 2 ?!
            quadrature_degree=quadrature_degree,
            w_incompressibility=1,
            elastic_behavior={"model":"H_dev", "parameters":material_parameters})
    else:
        problem = dmech.ElasticityProblem(
            mesh=mesh,
            compute_normals=1,
            boundaries_mf=boundaries_mf,
            u_degree=1,
            quadrature_degree=quadrature_degree,
            w_incompressibility=0,
            elastic_behavior={"model":"H", "parameters":material_parameters})

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
        dt_min=1.)

    if (load == "disp"):
        problem.add_constraint(
            V=problem.get_displacement_function_space().sub(0),
            sub_domains=boundaries_mf,
            sub_domain_id=xmax_id,
            val_ini=0.,
            val_fin=1.,
            k_step=k_step)
    elif (load == "volu"):
        problem.add_volume_force0_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim,
            F_fin=[1.]+[0.]*(dim-1),
            k_step=k_step)
    elif (load == "surf"):
        problem.add_surface_force0_loading_operator(
            measure=problem.dS(xmax_id),
            F_ini=[0.]*dim,
            F_fin=[1.]+[0.]*(dim-1),
            k_step=k_step)
    elif (load == "pres"):
        problem.add_pressure0_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=-0.,
            P_fin=-1.,
            k_step=k_step)
    elif (load == "pgra"):
        problem.add_pressure_gradient0_loading_operator(
            measure=problem.dS(),
            X0=[0.5]*dim,
            N0=[1.]+[0.]*(dim-1),
            P0_ini=-0.,
            P0_fin=-1.,
            DP_ini=-0.0,
            DP_fin=-0.5,
            k_step=k_step)
    elif (load == "tens"):
        problem.add_surface_tension0_loading_operator(
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
        stop_at_failure=0,
        clean_after_tests=1)

    dim_lst  = []
    dim_lst += [2]
    dim_lst += [3]
    for dim in dim_lst:

        PS_lst  = []
        if (dim == 2):
            PS_lst += [0]
            PS_lst += [1]
        elif (dim == 3):
            PS_lst += [0]
        for PS in PS_lst:

            incomp_lst  = []
            if (PS == 0):
                incomp_lst += [0]
                incomp_lst += [1]
            elif (PS == 1):
                incomp_lst += [0]
            for incomp in incomp_lst:

                load_lst  = []
                load_lst += ["disp"]
                load_lst += ["volu"]
                load_lst += ["surf"]
                load_lst += ["pres"]
                load_lst += ["pgra"]
                load_lst += ["tens"]
                for load in load_lst:

                    print("dim =",dim)
                    if (dim == 2): print("PS =",PS)
                    print("incomp =",incomp)
                    print("load =",load)

                    res_basename  = sys.argv[0][:-3]
                    res_basename += "-dim="+str(dim)
                    if (dim == 2): res_basename += "-PS"*PS + "-PE"*(1-PS)
                    res_basename += "-incomp="+str(incomp)
                    res_basename += "-load="+str(load)

                    test_elasticity(
                        dim=dim,
                        PS=PS,
                        incomp=incomp,
                        load=load,
                        res_basename=res_folder+"/"+res_basename,
                        verbose=0)

                    test.test(res_basename)
