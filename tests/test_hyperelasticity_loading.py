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

def test_hyperelasticity(
    dim,
    load, # disp, volu, surf, pres, pgra, tens
    res_basename,
    verbose=0):

    ################################################################### Mesh ###

    if   (dim==2):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.init_Rivlin_cube(dim=dim)
    elif (dim==3):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = dmech.init_Rivlin_cube(dim=dim)

    ################################################################ Problem ###

    problem = dmech.HyperelasticityProblem(
        mesh=mesh,
        compute_normals=1,
        boundaries_mf=boundaries_mf,
        U_degree=1,
        quadrature_degree="default",
        w_incompressibility=0,
        elastic_behavior={"model":"CGNHMR", "parameters":{"E":1., "nu":0.3}})

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
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=-0.0,
            P_fin=-0.5,
            k_step=k_step)
    elif (load == "pres"):
        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=-0.0,
            P_fin=-0.5,
            k_step=k_step)
    elif (load == "pgra0"):
        problem.add_surface_pressure_gradient0_loading_operator(
            measure=problem.dS(),
            X0_val=[0.5]*dim,
            N0_val=[1.]+[0.]*(dim-1),
            P0_ini=-0.0,
            P0_fin=-0.5,
            DP_ini=-0.00,
            DP_fin=-0.25,
            k_step=k_step)
    elif (load == "pgra"):
        problem.add_surface_pressure_gradient_loading_operator(
            measure=problem.dS(),
            X0_val=[0.5]*dim,
            N0_val=[1.]+[0.]*(dim-1),
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

    ################################################################# Solver ###

    solver = dmech.NonlinearSolver(
        problem=problem,
        parameters={
            "sol_tol":[1e-6],
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

        load_lst  = []
        load_lst += ["disp"]
        load_lst += ["volu0"]
        load_lst += ["volu"]
        load_lst += ["surf0"]
        load_lst += ["surf"]
        load_lst += ["pres0"]
        load_lst += ["pres"]
        load_lst += ["pgra0"]
        load_lst += ["pgra"]
        load_lst += ["tens"]
        for load in load_lst:

            print("dim =",dim)
            print("load =",load)

            res_basename  = sys.argv[0][:-3]
            res_basename += "-dim="+str(dim)
            res_basename += "-load="+str(load)

            test_hyperelasticity(
                dim=dim,
                load=load,
                res_basename=res_folder+"/"+res_basename,
                verbose=0)

            test.test(res_basename)
