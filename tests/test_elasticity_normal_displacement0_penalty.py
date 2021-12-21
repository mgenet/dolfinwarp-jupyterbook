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

def test_normal_displacement_penalty(
    dim,
    PS,
    incomp,
    decoup,
    load, # disp, volu, surf, pres
    res_basename,
    verbose=0):

    ################################################################### Mesh ###

    LX = 1.
    LY = 1.
    if (dim==3): LZ = 1.

    l = 1.
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

    ###################################################### Material behavior ###

    material_parameters = {
        "E":1.,
        "nu":0.5*(incomp)+0.3*(1-incomp)}

    if (incomp):
        hooke_dev = dmech.HookeDevElasticMaterial(
            parameters=material_parameters)
    else:
        if (decoup):
            hooke_dev = dmech.HookeDevElasticMaterial(
                parameters=material_parameters)
            hooke_bulk = dmech.HookeBulkElasticMaterial(
                dim=dim,
                parameters=material_parameters,
                PS=PS)
        else:
            hooke = dmech.HookeElasticMaterial(
                parameters=material_parameters,
                dim=dim,
                PS=PS)

    ################################################################ Problem ###

    if (incomp):
        problem = dmech.ElasticityProblem(
            mesh=mesh,
            compute_normals=1,
            boundaries_mf=boundaries_mf,
            U_degree=2, # MG20211219: Incompressibility requires U_degree >= 2 ?!
            quadrature_degree="default",
            w_incompressibility=1,
            elastic_behavior_dev=hooke_dev)
    else:
        if (decoup):
            problem = dmech.ElasticityProblem(
                mesh=mesh,
                compute_normals=1,
                boundaries_mf=boundaries_mf,
                U_degree=1,
                quadrature_degree="default",
                w_incompressibility=0,
                elastic_behavior_dev=hooke_dev,
                elastic_behavior_bulk=hooke_bulk)
        else:
            problem = dmech.ElasticityProblem(
                mesh=mesh,
                compute_normals=1,
                boundaries_mf=boundaries_mf,
                U_degree=1,
                quadrature_degree="default",
                w_incompressibility=0,
                elastic_behavior=hooke)

    ########################################## Boundary conditions & Loading ###

    problem.add_normal_displacement0_penalty_operator(
        measure=problem.dS(xmin_id),
        pen=1.)
    problem.add_normal_displacement0_penalty_operator(
        measure=problem.dS(ymin_id),
        pen=1.)
    if (dim == 3): problem.add_normal_displacement0_penalty_operator(
        measure=problem.dS(zmin_id),
        pen=1.)

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

            test_normal_displacement_penalty(
                dim=dim,
                PS=0,
                incomp=incomp,
                decoup=0,
                load="disp",
                res_basename=res_folder+"/"+res_basename,
                verbose=0)

            test.test(res_basename)
