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
    incomp,
    mat,
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

    mat_params = {
        "E":1.,
        "nu":0.5*(incomp)+0.3*(1-incomp),
        "dim":dim} # MG20220322: Necessary to compute correct bulk modulus

    if (incomp):
        problem = dmech.HyperelasticityProblem(
            mesh=mesh,
            compute_normals=1,
            boundaries_mf=boundaries_mf,
            U_degree=2, # MG20211219: Incompressibility requires U_degree >= 2 ?!
            quadrature_degree=quadrature_degree,
            w_incompressibility=1,
            elastic_behavior={"model":mat, "parameters":mat_params})
    else:
        problem = dmech.HyperelasticityProblem(
            mesh=mesh,
            compute_normals=1,
            boundaries_mf=boundaries_mf,
            U_degree=1,
            quadrature_degree=quadrature_degree,
            w_incompressibility=0,
            elastic_behavior={"model":mat, "parameters":mat_params})

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

    problem.add_constraint(
        V=problem.get_displacement_function_space().sub(0),
        sub_domains=boundaries_mf,
        sub_domain_id=xmax_id,
        val_ini=0.0,
        val_fin=0.5,
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

            mat_lst  = []
            if (incomp):
                mat_lst += ["NH"]
                # if (dim == 3): mat_lst += ["NH_bar"]
                mat_lst += ["NHMR"]
                # if (dim == 3): mat_lst += ["NHMR_bar"]
                mat_lst += ["SVK_dev"]
            else:
                mat_lst += ["CGNH"]
                # if (dim == 3): mat_lst += ["CGNH_bar"]
                mat_lst += ["CGNHMR"]
                # if (dim == 3): mat_lst += ["CGNHMR_bar"]
                mat_lst += ["SVK"]
            for mat in mat_lst:

                print("dim =",dim)
                print("incomp =",incomp)
                print("mat =",mat)

                res_basename  = sys.argv[0][:-3]
                res_basename += "-dim="+str(dim)
                res_basename += "-incomp="+str(incomp)
                res_basename += "-mat="+str(mat)

                test_hyperelasticity(
                    dim=dim,
                    incomp=incomp,
                    mat=mat,
                    res_basename=res_folder+"/"+res_basename,
                    verbose=0)

                test.test(res_basename)
