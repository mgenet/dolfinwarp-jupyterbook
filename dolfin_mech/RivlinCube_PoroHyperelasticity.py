#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import matplotlib.pyplot as mpl
import pandas
import numpy

import copy
import dolfin_mech as dmech

################################################################################

def RivlinCube_PoroHyperelasticity(
        dim=3,
        inverse=0,
        cube_params={},
        porosity_params={},
        mat_params={},
        step_params={},
        load_params={},
        move = {},
        res_basename="RivlinCube_PoroHyperelasticity",
        plot_curves=False,
        test = False,
        verbose=0,
        BC=1):

    ################################################################### Mesh ###

    if   (dim==2):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.RivlinCube_Mesh(dim=dim, params=cube_params)
    elif (dim==3):
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = dmech.RivlinCube_Mesh(dim=dim, params=cube_params)

    # coor = mesh.coordinates()
    # print(coor)

    if move.get("move", False) == True :
        Umove = move.get("U")
        dolfin.ALE.move(mesh, Umove)
        # coor = mesh.coordinates()
        # print("moved",coor)



    domains = dolfin.MeshFunction("size_t", mesh, dim)
    domains.set_all(1)

    V = dolfin.Measure(
            "dx",
            domain=mesh)

    ################################################################ Porosity ###

    porosity_type = porosity_params.get("type", "constant")
    porosity_val  = porosity_params.get("val", 0.5)

    if (porosity_type == "constant"):
        porosity_fun = None
    elif (porosity_type.startswith("mesh_function")):
        if (porosity_type == "mesh_function_constant"):
            porosity_mf = dolfin.MeshFunction(
                value_type="double",
                mesh=mesh,
                dim=dim,
                value=porosity_val)
        elif (porosity_type == "mesh_function_xml"):
            porosity_filename = res_basename+"-poro.xml"
            n_cells = len(mesh.cells())
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <mesh_function type="double" dim="'+str(dim)+'" size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    file.write('    <entity index="'+str(k_cell)+'" value="'+str(porosity_val)+'"/>\n')
                file.write('  </mesh_function>\n')
                file.write('</dolfin>\n')
                file.close()
            porosity_mf = dolfin.MeshFunction(
                "double",
                mesh,
                porosity_filename)
        elif (porosity_type == "mesh_function_xml_custom"):
            porosity_filename = res_basename+"-poro.xml"
            n_cells = len(mesh.cells())
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <mesh_function type="double" dim="'+str(dim)+'" size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    file.write('    <entity index="'+str(k_cell)+'" value="'+str(porosity_val[k_cell])+'"/>\n')
                file.write('  </mesh_function>\n')
                file.write('</dolfin>\n')
                file.close()
            porosity_mf = dolfin.MeshFunction(
                "double",
                mesh,
                porosity_filename)
        porosity_expr = dolfin.CompiledExpression(getattr(dolfin.compile_cpp_code(dmech.get_ExprMeshFunction_cpp_pybind()), "MeshExpr")(), mf=porosity_mf, degree=0)
        porosity_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        porosity_fun = dolfin.interpolate(porosity_expr, porosity_fs)
        porosity_val = None
    elif (porosity_type.startswith("function")):
        porosity_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        if (porosity_type == "function_constant"):
            porosity_fun = dolfin.Function(porosity_fs)
            porosity_fun.vector()[:] = porosity_val
        elif (porosity_type == "function_xml"):
            porosity_filename = res_basename+"-poro.xml"
            n_cells = len(mesh.cells())
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <function_data size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    file.write('    <dof index="'+str(k_cell)+'" value="'+str(porosity_val)+'" cell_index="'+str(k_cell)+'" cell_dof_index="0"/>\n')
                file.write('  </function_data>\n')
                file.write('</dolfin>\n')
                file.close()
            porosity_fun = dolfin.Function(
                porosity_fs,
                porosity_filename)
        porosity_val = None


    ################################################################ Problem ###

    # quadrature_degree = "default"
    quadrature_degree = 4

    if (inverse):
        problem = dmech.InversePoroHyperelasticityProblem(
            mesh=mesh,
            define_facet_normals=1,
            boundaries_mf=boundaries_mf,
            displacement_degree=1,
            quadrature_degree=quadrature_degree,
            porosity_init_val=porosity_val,
            porosity_init_fun=porosity_fun,
            skel_behavior=mat_params,
            bulk_behavior=mat_params,
            pore_behavior=mat_params)
    else:
        problem = dmech.PoroHyperelasticityProblem(
            mesh=mesh,
            define_facet_normals=1,
            boundaries_mf=boundaries_mf,
            displacement_degree=1,
            quadrature_degree=quadrature_degree,
            porosity_init_val=porosity_val,
            porosity_init_fun=porosity_fun,
            skel_behavior=mat_params,
            bulk_behavior=mat_params,
            pore_behavior=mat_params)


    ########################################## Boundary conditions & Loading ###

    # problem.add_constraint(V=problem.get_displacement_function_space().sub(0), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=0.)
    # problem.add_constraint(V=problem.get_displacement_function_space().sub(1), sub_domains=boundaries_mf, sub_domain_id=ymin_id, val=0.)
    # if (dim==3):
    #     problem.add_constraint(V=problem.get_displacement_function_space().sub(2),sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.)

    # print("dolfin.assemble(dolfin.Constant(1.) * problem.dS(xmin_id)):"+str(dolfin.assemble(dolfin.Constant(1.) * problem.dS(xmin_id))))

    # if BC ==1:
    #     problem.add_constraint(
    #             V=problem.get_displacement_function_space(), 
    #             val=[0.]*dim,
    #             sub_domain=boundaries_point,
    #             sub_domain_id = pinpoint1_id,
    #             method='pointwise')
    #     problem.add_constraint(
    #             V=problem.get_displacement_function_space().sub(1), 
    #             val=0.,
    #             sub_domain=boundaries_point,
    #             sub_domain_id=pinpoint2_id,
    #             method='pointwise')
    #     problem.add_constraint(
    #             V=problem.get_displacement_function_space().sub(2), 
    #             val=0.,
    #             sub_domain=boundaries_point,
    #             sub_domain_id=pinpoint2_id,
    #             method='pointwise')
    #     problem.add_constraint(
    #             V=problem.get_displacement_function_space().sub(2), 
    #             val=0.,
    #             sub_domain=boundaries_point,
    #             sub_domain_id=pinpoint3_id,
    #             method='pointwise')

    # coor = mesh.coordinates()
    # print(inverse)
    # print("/n")
    # print(coor)


    if BC ==1:
        # if move.get("move", False) == True:
        #     pinpoint_sd = dmech.PinpointSubDomain(coords=[0.,0.,0.], tol=1e-3)
        # else :
        pinpoint_sd = dmech.PinpointSubDomain(coords=[0.,0.,0.], tol=1e-4)
        problem.add_constraint(
                V=problem.get_displacement_function_space(), 
                val=[0.] * dim,
                sub_domain=pinpoint_sd,
                method='pointwise')
        if move.get("move", False) == False:
            pinpoint_sd = dmech.PinpointSubDomain(coords=[cube_params["X1"],0.,0.], tol=1e-4)
        else :
            pinpoint_sd = dmech.PinpointSubDomain(coords=[cube_params["X1"] + Umove([cube_params["X1"], 0., 0.])[0],0.,0.], tol=1e-4)
        # pinpoint_sd = dmech.PinpointSubDomain(coords=[cube_params["X1"],0.,0.], tol=1e-3)
        problem.add_constraint(
                V=problem.get_displacement_function_space().sub(1), 
                val=0.,
                sub_domain=pinpoint_sd,
                method='pointwise')
        problem.add_constraint(
                V=problem.get_displacement_function_space().sub(2), 
                val=0.,
                sub_domain=pinpoint_sd,
                method='pointwise')
        if move.get("move", False) == False:
            pinpoint_sd = dmech.PinpointSubDomain(coords=[0.,cube_params["Y1"],0.], tol=1e-4)
        else :
            pinpoint_sd = dmech.PinpointSubDomain(coords=[0. + Umove([0., cube_params["Y1"], 0.])[0],cube_params["Y1"] + Umove([0., cube_params["Y1"], 0.])[1],0.], tol=1e-3)
        # pinpoint_sd = dmech.PinpointSubDomain(coords=[0.,cube_params["Y1"],0.], tol=1e-3)
        problem.add_constraint(
                V=problem.get_displacement_function_space().sub(2), 
                val=0.,
                sub_domain=pinpoint_sd,
                method='pointwise')
    # elif BC ==2:
    #     pinpoint_sd = dmech.PinpointSubDomain(coords=[0.,0.,Z1], tol=1e-3)
    #     problem.add_constraint(
    #         V=problem.get_displacement_function_space(), 
    #         val=[0.]*dim,
    #         sub_domain=pinpoint_sd,
    #         method='pointwise')
    #     pinpoint_sd = dmech.PinpointSubDomain(coords=[X1,0.,Z1], tol=1e-3)
    #     problem.add_constraint(
    #         V=problem.get_displacement_function_space().sub(1), 
    #         val=0.,
    #         sub_domain=pinpoint_sd,
    #         method='pointwise')
    #     problem.add_constraint(
    #         V=problem.get_displacement_function_space().sub(2), 
    #         val=0.,
    #         sub_domain=pinpoint_sd,
    #         method='pointwise')
    #     pinpoint_sd = dmech.PinpointSubDomain(coords=[0.,Y1,Z1], tol=1e-3)
    #     problem.add_constraint(
    #         V=problem.get_displacement_function_space().sub(2), 
    #         val=0.,
    #         sub_domain=pinpoint_sd,
    #         method='pointwise')

    Deltat = step_params.get("Deltat", 1.)
    dt_ini = step_params.get("dt_ini", 1.)
    dt_min = step_params.get("dt_min", 1.)
    dt_max = step_params.get("dt_max", 1.)
    k_step = problem.add_step(
        Deltat=Deltat,
        dt_ini=dt_ini,
        dt_min=dt_min,
        dt_max=dt_max)

    load_type = load_params.get("type", "internal")
    if (load_type == "internal"):
        pf = load_params.get("pf", +0.5)
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0., pf_fin=pf,
            k_step=k_step)
    elif (load_type == "external"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0., pf_fin=0.,
            k_step=k_step)
        P = load_params.get("P", -0.5)
        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=0., P_fin=P,
            k_step=k_step)
        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(ymax_id),
            P_ini=0., P_fin=P,
            k_step=k_step)
        if (dim==3): problem.add_surface_pressure_loading_operator(
            measure=problem.dS(zmax_id),
            P_ini=0., P_fin=P,
            k_step=k_step)
    elif (load_type == "external0"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0., pf_fin=0.,
            k_step=k_step)
        P = load_params.get("P", -0.5)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=0., P_fin=P,
            k_step=k_step)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(ymax_id),
            P_ini=0., P_fin=P,
            k_step=k_step)
        if (dim==3): problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(zmax_id),
            P_ini=0., P_fin=P,
            k_step=k_step)
    elif (load_type == "volu0"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        f = load_params.get("f", 0.5)
        problem.add_volume_force0_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim,
            F_fin = [0.]*(dim-1) + [f],
            k_step=k_step)
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        P = load_params.get("P", -0.5)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=0,
            P_fin=P,
            k_step=k_step)
        problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(ymax_id),
            P_ini=0,
            P_fin=P,
            k_step=k_step)
        if (dim==3): problem.add_surface_pressure0_loading_operator(
            measure=problem.dS(zmax_id),
            P_ini=0,
            P_fin=P,
            k_step=k_step)
    elif (load_type == "volu"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        f = load_params.get("f", 1.)
        problem.add_volume_force_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim,
            F_fin=[0.]*(dim-1) + [f],
            k_step=k_step)
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        P = load_params.get("P", -0.5)
        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=0,
            P_fin=P,
            k_step=k_step)
        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(ymax_id),
            P_ini=0,
            P_fin=P,
            k_step=k_step)
        if (dim==3): problem.add_surface_pressure_loading_operator(
            measure=problem.dS(zmax_id),
            P_ini=0,
            P_fin=P,
            k_step=k_step)
    elif (load_type == "pgra0"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        f = load_params.get("f", 0.5)
        problem.add_volume_force0_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim,
            F_fin=[0.]*(dim-1)+[f],
            k_step=k_step)
        X0 = load_params.get("X0", [50]*dim)
        N0 = load_params.get("N0", [0.]*(dim-1)+[1.])
        P0 = load_params.get("P0", -0.5)
        DP = load_params.get("DP", -0.25)
        problem.add_surface_pressure_gradient0_loading_operator(
            measure=problem.dS,
            X0_val=X0,
            N0_val=N0,
            P0_ini=0.,
            P0_fin=P0,
            DP_ini=0.,
            DP_fin=DP,
            k_step=k_step)
    elif (load_type == "pgra"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        X0 = load_params.get("X0", [50]*dim)
        N0 = load_params.get("N0", [0.]*(dim-1)+[1.])
        P0 = load_params.get("P0", -0.5)
        DP = load_params.get("DP", None)
        F0 = load_params.get("F0", None)
        if DP == None and F0 == None:
            print("error, DP or F0 should be specified")
        if DP == None:
            problem.add_surface_pressure_gradient_loading_operator(
                measure=problem.dS(),
                X0_val=X0,
                N0_val=N0,
                P0_ini=0.,
                P0_fin=P0,
                DP_ini=0.,
                DP_fin=DP,
                F0 =  F0, 
                k_step=k_step)
        else:
            problem.add_surface_pressure_gradient_loading_operator(
                measure=problem.dS,
                X0_val=X0,
                N0_val=N0,
                P0_ini=0.,
                P0_fin=P0,
                DP_ini=0.,
                DP_fin=DP,
                F0 = None,
                k_step=k_step)
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        f = load_params.get("f", 1.)
        problem.add_volume_force0_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim,
            F_fin=[0.]*(dim-1) + [f],
            k_step=k_step)

    ################################################# Quantities of Interest ###

    problem.add_deformed_volume_qoi()
    problem.add_global_strain_qois()
    problem.add_global_stress_qois()
    problem.add_global_porosity_qois()
    problem.add_global_fluid_pressure_qoi()

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

    ################################################################## Plots ###

    if (plot_curves):
        qois_data = pandas.read_csv(
            res_basename+"-qois.dat",
            delim_whitespace=True,
            comment="#",
            names=open(res_basename+"-qois.dat").readline()[1:].split())

        qois_fig, qois_axes = mpl.subplots()
        all_strains = ["E_XX", "E_YY"]
        if (dim == 3): all_strains += ["E_ZZ"]
        all_strains += ["E_XY"]
        if (dim == 3): all_strains += ["E_YZ", "E_ZX"]
        qois_data.plot(x="t", y=all_strains, ax=qois_axes, ylabel="Green-Lagrange strain")
        qois_fig.savefig(res_basename+"-strains-vs-time.pdf")

        for comp in ["skel", "bulk", "tot"]:
            qois_fig, qois_axes = mpl.subplots()
            all_stresses = ["s_"+comp+"_XX", "s_"+comp+"_YY"]
            if (dim == 3): all_stresses += ["s_"+comp+"_ZZ"]
            all_stresses += ["s_"+comp+"_XY"]
            if (dim == 3): all_stresses += ["s_"+comp+"_YZ", "s_"+comp+"_ZX"]
            qois_data.plot(x="t", y=all_stresses, ax=qois_axes, ylabel="Cauchy stress")
            qois_fig.savefig(res_basename+"-stresses-"+comp+"-vs-time.pdf")

        qois_fig, qois_axes = mpl.subplots()
        all_porosities = []
        if (inverse):
            all_porosities += ["phis0", "phif0", "Phis0", "Phif0"]
        else:
            all_porosities += ["Phis", "Phif", "phis", "phif"]
        qois_data.plot(x="t", y=all_porosities, ax=qois_axes, ylim=[0,1], ylabel="porosity")
        qois_fig.savefig(res_basename+"-porosities-vs-time.pdf")

        qois_fig, qois_axes = mpl.subplots()
        qois_data.plot(x="pf", y=all_porosities, ax=qois_axes, ylim=[0,1], ylabel="porosity")
        qois_fig.savefig(res_basename+"-porosities-vs-pressure.pdf")

    for foi in problem.fois:
        if foi.name == "Phis0":
            phi = foi.func.vector().get_local()
 
    phi0 = 0.
    for qoi in problem.qois:
        if qoi.name == "Phis0":
            phi0 = qoi.value

    
    # dofs_all = problem.sol_fs.tabulate_dof_coordinates() #.reshape(problem.sol_fs.dim(),mesh.geometry().dim())

    # # dof_new = problem.get_subsol_function_space(name=problem.get_displacement_name()).dofmap().dofs()

    # x = dofs_all[:, 0 ]

    # indices = numpy.where(numpy.logical_and( x>-1,x < 101))[0]

    # xs = dofs_all[indices]


    # for coordinate in V0_dofs_x:
    #     if coordinate[0] == 0. and coordinate[1] == 0. and coordinate[2] == 0.:
    #         clamped.append(problem.get_subsols_func_lst()[0].vector().get_local()[i])
    #     i+=1

    # print(problem.get_subsol_function_space[0].vector()[dolfin.vertex_to_dof_map(problem.sol_fs)].array())

    # coor = mesh.coordinates()

    # U = []
    # print("for the case",BC)
    # for i in range(mesh.num_vertices()):
        # print([coor[i][0], coor[i][1], coor[i][2], problem.get_subsols_func_lst()[0](coor[i][0], coor[i][1], coor[i][2])])
        # U.append([coor[i][0],coor[i][1], coor[i][2]] + problem.get_subsols_func_lst()[0](coor[i][0], coor[i][1], coor[i][2]))
        # print([coor[i][0],coor[i][1], coor[i][2]] + problem.get_subsols_func_lst()[0](coor[i][0], coor[i][1], coor[i][2]))
    
    
    # print("new coordinates")
    # print("\n")
    # print(U)
    # print("\n")

    # if test:

    #     theta_x = numpy.arctan( (problem.get_subsols_func_lst()[0](0., cube_params["Y1"], 0.)[2]) / ( problem.get_subsols_func_lst()[0](0., cube_params["Y1"], 0.)[1]+ cube_params["Y1"] )) #/ 180 * 3.14
    #     theta_y = numpy.arctan( (problem.get_subsols_func_lst()[0](cube_params["X1"], 0., 0.)[2]) / (problem.get_subsols_func_lst()[0](cube_params["X1"], 0., 0.)[0] + cube_params["X1"])) #/ 180 * 3.14
    #     theta_z = numpy.arctan( (problem.get_subsols_func_lst()[0](cube_params["X1"], 0., 0.)[1])/(problem.get_subsols_func_lst()[0](cube_params["X1"], 0., 0.)[0]+ cube_params["X1"])) #/180 * 3.14

    #     Rx = numpy.array([[1.,0.,0.], [0., numpy.cos(theta_x), -numpy.sin(theta_x)], [0., numpy.sin(theta_x), numpy.cos(theta_x)]])

    #     Ry = numpy.array([[numpy.cos(theta_y),0.,numpy.sin(theta_y)], [0.,1.,0.], [-numpy.sin(theta_y), 0., numpy.cos(theta_y)]])

    #     Rz = numpy.array([[numpy.cos(theta_z),-numpy.sin(theta_z),0.], [numpy.sin(theta_z), numpy.cos(theta_z),0.], [0., 0., 1.]])
    
    #     return(problem.get_subsols_func_lst()[0], U, Rx, Ry, Rz, problem.get_subsols_func_lst()[0],  phi, phi0, V)

    # else:

    return( problem.get_subsols_func_lst()[0],  phi, phi0, V)


