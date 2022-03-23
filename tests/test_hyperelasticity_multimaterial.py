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
    res_basename,
    verbose=0):

    ################################################################### Mesh ###

    if   (dim==2):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.init_Rivlin_cube(dim=dim, l=0.1)
    elif (dim==3):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = dmech.init_Rivlin_cube(dim=dim, l=0.1)

    mat1_sd = dolfin.CompiledSubDomain("x[0] <= x0", x0=1/2)
    mat2_sd = dolfin.CompiledSubDomain("x[0] >= x0", x0=1/2)

    mat1_id = 1
    mat2_id = 2

    domains_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()) # MG20180418: size_t looks like unisgned int, but more robust wrt architecture and os
    domains_mf.set_all(0)
    mat1_sd.mark(domains_mf, mat1_id)
    mat2_sd.mark(domains_mf, mat2_id)

    ################################################################ Problem ###

    mat1_params = {
        "E":1.,
        "nu":0.5*(incomp)+0.3*(1-incomp)}

    mat2_params = {
        "E":10.,
        "nu":0.5*(incomp)+0.3*(1-incomp)}

    if (incomp):
        problem = dmech.HyperelasticityProblem(
            mesh=mesh,
            domains_mf=domains_mf,
            compute_normals=1,
            boundaries_mf=boundaries_mf,
            U_degree=2, # MG20211219: Incompressibility requires U_degree >= 2 ?!
            quadrature_degree="default",
            w_incompressibility=1,
            elastic_behaviors=[
                {"subdomain_id":mat1_id, "model":"NHMR", "parameters":mat1_params, "suffix":"1"},
                {"subdomain_id":mat2_id, "model":"NHMR", "parameters":mat2_params, "suffix":"2"}])
    else:
        problem = dmech.HyperelasticityProblem(
            mesh=mesh,
            domains_mf=domains_mf,
            compute_normals=1,
            boundaries_mf=boundaries_mf,
            U_degree=1,
            quadrature_degree="default",
            w_incompressibility=0,
            elastic_behaviors=[
                {"subdomain_id":mat1_id, "model":"CGNHMR", "parameters":mat1_params, "suffix":"1"},
                {"subdomain_id":mat2_id, "model":"CGNHMR", "parameters":mat2_params, "suffix":"2"}])

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

            print("dim =",dim)
            print("incomp =",incomp)

            res_basename  = sys.argv[0][:-3]
            res_basename += "-dim="+str(dim)
            res_basename += "-incomp="+str(incomp)

            test_hyperelasticity_multimaterial(
                dim=dim,
                incomp=incomp,
                res_basename=res_folder+"/"+res_basename,
                verbose=0)

            test.test(res_basename)
