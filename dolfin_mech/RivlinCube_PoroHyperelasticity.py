#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################
# from copy import deepcopy
import dolfin
import matplotlib.pyplot as mpl
import pandas 
import numpy
import copy

from FIAT.reference_element import UFCTetrahedron, UFCTriangle
from FIAT import ufc_simplex, Lagrange

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
        verbose=0,
        inertia_val = 1e-5,
        multimaterial = 0,
        get_results_fields = 0,
        mesh_from_file=0,
        BC=1):

    ################################################################### Mesh ###

    if mesh_from_file:
        mesh = dolfin.Mesh()
        dolfin.XDMFFile("/Users/peyrault/Documents/Gravity/Tests/Meshes_and_poro_files/Zygot.xdmf").read(mesh)
        # mesh = dolfin.refine(mesh)
        # mesh = dolfin.refine(mesh)
        boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # MG20180418: size_t looks like unisgned int, but more robust wrt architecture and os
        boundaries_mf.set_all(0)
    else:
        if   (dim==2):
            mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.RivlinCube_Mesh(dim=dim, params=cube_params)
        elif (dim==3):
            mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = dmech.RivlinCube_Mesh(dim=dim, params=cube_params)


    if mesh_from_file:
        if multimaterial :
            ymin = mesh.coordinates()[:, 1].min()
            ymax = mesh.coordinates()[:, 1].max()
            delta_y = ymax - ymin
            domains_mf = None
            tol = 1E-14
            number_zones = len(mat_params)
            length_zone = delta_y*0.1
            domains_mf = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim())
            domains_mf.set_all(0)
            subdomain_lst = []
            subdomain_lst.append(dolfin.CompiledSubDomain("x[1] <= y1 + tol",  y1=ymin+length_zone, tol=tol))
            subdomain_lst[0].mark(domains_mf, 0)
            for mat_id in range(0, number_zones):
                subdomain_lst.append(dolfin.CompiledSubDomain(" x[1] >= y1 - tol",  y1=ymin+length_zone*(mat_id+1), tol=tol))
                subdomain_lst[mat_id].mark(domains_mf, mat_id)
                mat_params[mat_id]["subdomain_id"] = mat_id
            # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Gravity_cluster/Tests/boundaries.pvd") 
            # boundary_file << domains_mf
    else:
        if multimaterial :
            domains_mf = None
            # print(len(mat_params))
            if len(mat_params)>=2 :
                tol = 1E-14
                number_zones = len(mat_params)
                mid_point = abs(cube_params.get("Y1", 1.)-cube_params.get("Y0", 0.))/number_zones
                domains_mf = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim())
                if number_zones == 2:
                    domains_mf.set_all(0)
                    subdomain_0 = dolfin.CompiledSubDomain("x[1] <= y1 + tol",  y1=mid_point, tol=tol)
                    subdomain_1 = dolfin.CompiledSubDomain(" x[1] >= y1 - tol",  y1=mid_point, tol=tol)
                    subdomain_0.mark(domains_mf, 0)
                    mat_params[0]["subdomain_id"] = 0
                    subdomain_1.mark(domains_mf, 1)
                    mat_params[1]["subdomain_id"] = 1
                    # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Gravity_cluster/Tests/boundaries.pvd") 
                    # boundary_file << domains_mf
                elif number_zones == 3 :
                    domains_mf.set_all(0)
                    # print("creating 3 subdomains...")
                    y1 = mid_point
                    y2 = 2*mid_point
                    # print("y1, y2", y1, y2)
                    subdomain_0 = dolfin.CompiledSubDomain("x[1] <= y1 + tol",  y1=y1, tol=tol)
                    subdomain_1 = dolfin.CompiledSubDomain(" x[1] >= y1 - tol",  y1=y1, tol=tol)
                    subdomain_2 = dolfin.CompiledSubDomain("x[1] >= y2 - tol",  y2=y2, tol=tol)
                    subdomain_0.mark(domains_mf, 0)
                    mat_params[0]["subdomain_id"] = 0
                    subdomain_1.mark(domains_mf, 1)
                    mat_params[1]["subdomain_id"] = 1
                    subdomain_2.mark(domains_mf, 2)
                    mat_params[2]["subdomain_id"] = 2
                    # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Gravity_cluster/Tests/boundaries.pvd") 
                    # boundary_file << domains_mf
                else:
                    y1 = mid_point
                    domains_mf.set_all(0)
                    subdomain_lst = []
                    subdomain_lst.append(dolfin.CompiledSubDomain("x[1] <= y1 + tol",  y1=y1, tol=tol))
                    subdomain_lst[0].mark(domains_mf, 0)
                    for mat_id in range(0, number_zones):
                        subdomain_lst.append(dolfin.CompiledSubDomain(" x[1] >= y1 - tol",  y1=(mat_id+1)*y1, tol=tol))
                        subdomain_lst[mat_id].mark(domains_mf, mat_id)
                        mat_params[mat_id]["subdomain_id"] = mat_id
                    # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Gravity_cluster/Tests/boundaries.pvd") 
                    # boundary_file << domains_mf

  
    
    mesh.init()
    # bmesh = dolfin.BoundaryMesh(mesh, 'exterior')
    # dolfin.XDMFFile("/Users/peyrault/Documents/Gravity/Tests/Article/boundary_mesh.xdmf").write(bmesh)
    # facet_domains = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    # facet_domains.set_all(0)
    # for f in dolfin.facets(mesh):
    #     if f.exterior():
    #     # if any(ff.exterior() for ff in dolfin.facets(f)):
    #         facet_domains[f] = 1

    vertex_coords = []
    facet_domains = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    facet_domains.set_all(0)
    # f_on_boundary = dolfin.CompiledSubDomain("on_boundary ")
    for f in dolfin.facets(mesh):
        if f.exterior():
            facet_domains[f] = 1
            for vertex in dolfin.vertices(f):
                vertex_coords.append(list(vertex.point().array()))

    for f in dolfin.facets(mesh):
        # for vertex in dolfin.vertices(f):
        #     print(list(vertex.point().array()) in vertex_coords)
            # print(list(vertex.point().array()))
        if any(list(vertex.point().array()) in vertex_coords for vertex in dolfin.vertices(f)):
            # print("marking facet")
            facet_domains[f] = 1

        # for v in dolfin.vertices(f):
        # if any(ff.exterior() for ff in dolfin.facets(f)):
                # facet_domains[f] = 1

    # print("vertex_coords", vertex_coords)
    

    boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Tests/Article/boundaries_facets.pvd") 
    boundary_file << facet_domains

    

    # cells_domains = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    # boundary_adjacent_cells = [c for c in dolfin.cells(mesh) 
    #                          if any(f.exterior()  for f in dolfin.facets(c))]
    
    # for c in boundary_adjacent_cells:
    #     cells_domains[c] = 1

    # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Tests/Article/boundaries_cells.pvd") 
    # boundary_file << cells_domains
    


    domains_dirichlet = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    domains_dirichlet.set_all(0)
    # external = dolfin.CompiledSubDomain("x[0] >= 100.0 - DOLFIN_EPS or x[0] < DOLFIN_EPS or x[1] < DOLFIN_EPS or x[2] < DOLFIN_EPS or x[1] >= 100.0 - DOLFIN_EPS or x[2] >= 100.0 - DOLFIN_EPS")
    # internal = dolfin.CompiledSubDomain(" 100.0 > x[0]  && x[0]>0. &&  100.0 > x[1]  && x[1]> 0. && 100.0 > x[2]  && x[2]> 0.")  
    internal = dolfin.CompiledSubDomain("not on_boundary ")
    external = dolfin.CompiledSubDomain("on_boundary ")
    external_id = 1
    external.mark(domains_dirichlet, external_id)
    # internal_id = 1
    # internal.mark(domains_dirichlet, internal_id)
    # internal.mark(boundaries_mf, internal_id)
    boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Tests/Article/boundaries.pvd") 
    boundary_file << domains_dirichlet

   

    


    
    ################################################################ Porosity ###

    porosity_type = porosity_params.get("type", "constant")
    porosity_val  = porosity_params.get("val", 0.5)

    if (porosity_type=="from_file"):
        porosity_filename = porosity_val
        porosity_mf = dolfin.MeshFunction(
            "double",
            mesh,
            porosity_filename)
        porosity_expr = dolfin.CompiledExpression(getattr(dolfin.compile_cpp_code(dmech.get_ExprMeshFunction_cpp_pybind()), "MeshExpr")(), mf=porosity_mf, degree=0)
        porosity_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        porosity_fun = dolfin.interpolate(porosity_expr, porosity_fs)
        porosity_val = None
    if (porosity_type == "constant"):
        porosity_fun = None
        # print("constant")
    elif (porosity_type.startswith("mesh_function")):
        # print("mesh_function")
        if (porosity_type == "mesh_function_constant"):
            # print("mesh_function constant")
            porosity_mf = dolfin.MeshFunction(
                value_type="double",
                mesh=mesh,
                dim=dim,
                value=porosity_val)
        elif (porosity_type == "mesh_function_xml"):
            # print("mesh_function xml")
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
        elif (porosity_type == "mesh_function_random_xml"):
            # print("mesh_function xml")
            porosity_filename = res_basename+"-poro.xml"
            n_cells = len(mesh.cells())
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <mesh_function type="double" dim="'+str(dim)+'" size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    value = numpy.random.uniform(low=0.4, high=0.6)
                    # positive_value = False
                    # while not positive_value:
                        # value = float(numpy.random.normal(loc=0.5, scale=0.13, size=1)[0])
                        # if value >0 and value<1:
                            # positive_value = True
                    file.write('    <entity index="'+str(k_cell)+'" value="'+str(value)+'"/>\n')
                file.write('  </mesh_function>\n')
                file.write('</dolfin>\n')
                file.close()
            porosity_mf = dolfin.MeshFunction(
                "double",
                mesh,
                porosity_filename)
        elif (porosity_type == "mesh_function_xml_custom"):
            # print("mesh_function xml custom")
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
        # print("function")
        porosity_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        if (porosity_type == "function_constant"):
            # print("function constant")
            porosity_fun = dolfin.Function(porosity_fs)
            porosity_fun.vector()[:] = porosity_val
        elif (porosity_type == "function_xml"):
            # print("function xml")
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
        elif (porosity_type == "function_xml_custom"):
            # print("function xml")
            porosity_filename = res_basename+"-poro.xml"
            # print("porosity_filename=", porosity_filename)
            n_cells = len(mesh.cells())
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <function_data size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    # print("kcell=", k_cell)
                    # print("porosity for kcell", porosity_val[k_cell])
                    file.write('    <dof index="'+str(k_cell)+'" value="'+str(porosity_val[k_cell])+'" cell_index="'+str(k_cell)+'" cell_dof_index="0"/>\n')
                file.write('  </function_data>\n')
                file.write('</dolfin>\n')
                file.close()
            porosity_fun = dolfin.Function(
                porosity_fs,
                porosity_filename)
            porosity_val = None
    
    # print("porosity defined")


    if move.get("move", False) == True :
        Umove = move.get("U")
        # print("displacement field retrived")
        dolfin.ALE.move(mesh, Umove)
        # print("mesh moved")

    V = dolfin.Measure(
            "dx",
            domain=mesh)

    # boundary_mesh = dolfin.BoundaryMesh(mesh, "exterior")
    # print("deformed volume is", dolfin.assemble(1*V))
    
    ################################################################ Problem ###    
    if(multimaterial):
        if (inverse):
            problem = dmech.InversePoroHyperelasticityProblem(
                inverse=1,
                mesh=mesh,
                define_facet_normals=1,
                boundaries_mf=boundaries_mf,
                domains_mf = domains_mf,
                displacement_degree=1,
                # quadrature_degree = 10,
                porosity_init_val=porosity_val,
                porosity_init_fun=porosity_fun,
                # skel_behavior=mat_params,
                skel_behaviors=mat_params,
                # bulk_behavior=mat_params,
                bulk_behaviors=mat_params,
                # pore_behavior=mat_params,
                pore_behaviors=mat_params)
        else:
            problem = dmech.PoroHyperelasticityProblem(
                inverse=0,
                mesh=mesh,
                define_facet_normals=1,
                boundaries_mf=boundaries_mf,
                domains_mf = domains_mf,
                displacement_degree=1,
                # quadrature_degree = "default",
                porosity_init_val=porosity_val,
                porosity_init_fun=porosity_fun,
                # skel_behavior=mat_params,
                skel_behaviors=mat_params,
                # bulk_behavior=mat_params,
                bulk_behaviors=mat_params,
                # pore_behavior=mat_params,
                pore_behaviors=mat_params)
    else:
        if (inverse):
            problem = dmech.InversePoroHyperelasticityProblem(
                inverse=1,
                mesh=mesh,
                # boundary_mesh=boundary_mesh,
                define_facet_normals=1,
                boundaries_mf=boundaries_mf,
                # domains_mf = domains_mf,
                # domains_dirichlet=domains_dirichlet,
                displacement_degree=2,
                quadrature_degree = "default",
                porosity_init_val=porosity_val,
                porosity_init_fun=porosity_fun,
                skel_behavior=mat_params,
                # skel_behaviors=mat_params,
                bulk_behavior=mat_params,
                # bulk_behaviors=mat_params,
                pore_behavior=mat_params)
                # pore_behaviors=mat_params)
        else:
            problem = dmech.PoroHyperelasticityProblem(
                inverse=0,
                mesh=mesh,
                # boundary_mesh=boundary_mesh,
                define_facet_normals=1,
                boundaries_mf=boundaries_mf,
                # domains_mf = domains_mf,
                displacement_degree=2,
                quadrature_degree = "default",
                porosity_init_val=porosity_val,
                porosity_init_fun=porosity_fun,
                skel_behavior=mat_params,
                # skel_behaviors=mat_params,
                bulk_behavior=mat_params,
                # bulk_behaviors=mat_params,
                pore_behavior=mat_params)
                # pore_behaviors=mat_params)


    ########################################## Boundary conditions & Loading ###
    # problem.add_constraint(V=problem.get_displacement_function_space().sub(0), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=0.)
    # problem.add_constraint(V=problem.get_displacement_function_space().sub(1), sub_domains=boundaries_mf, sub_domain_id=ymin_id, val=0.)
    # if (dim==3):
    #     problem.add_constraint(V=problem.get_displacement_function_space().sub(0), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.)
    #     problem.add_constraint(V=problem.get_displacement_function_space().sub(1), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.)
    #     problem.add_constraint(V=problem.get_displacement_function_space().sub(2), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.)
    
    
    # if not inverse:
    #     print("Phis0", problem.Phis0.vector()[:])
    #     print("rho0", dolfin.assemble(problem.Phis0*V)/dolfin.assemble(1*V))


    BC=0
    if BC == 1:
        pinpoint_sd = dmech.PinpointSubDomain(coords=[0.,0.,0.], tol=1e-4)
        problem.add_constraint(
                V=problem.get_displacement_function_space(), 
                val=[0.] * dim,
                sub_domain=pinpoint_sd,
                method='pointwise')
        if move.get("move", False) == False:
            pinpoint_sd = dmech.PinpointSubDomain(coords=[cube_params["X1"],0.,0.], tol=1e-4)
        else :
            pinpoint_sd = dmech.PinpointSubDomain(coords=[cube_params["X1"] + Umove([cube_params["X1"], 0., 0.])[0], Umove([cube_params["X1"], 0., 0.])[1], Umove([cube_params["X1"], 0., 0.])[2]], tol=1e-4)
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
            pinpoint_sd = dmech.PinpointSubDomain(coords=[0. + Umove([0., cube_params["Y1"], 0.])[0],cube_params["Y1"] + Umove([0., cube_params["Y1"], 0. ])[1], Umove([cube_params["X1"], 0., 0.])[2]], tol=1e-3)
        problem.add_constraint(
                V=problem.get_displacement_function_space().sub(2), 
                val=0.,
                sub_domain=pinpoint_sd,
                method='pointwise')
    elif BC ==2:
        pinpoint_sd = dmech.PinpointSubDomain(coords=[0., 0., cube_params["Z1"]], tol=1e-3)
        problem.add_constraint(
            V=problem.get_displacement_function_space(), 
            val=[0.]*dim,
            sub_domain=pinpoint_sd,
            method='pointwise')
        if move.get("move", False) == False:
            pinpoint_sd = dmech.PinpointSubDomain(coords=[cube_params["X1"], 0., cube_params["Z1"]], tol=1e-3)
        else :
            pinpoint_sd = dmech.PinpointSubDomain(coords=[cube_params["X1"] + Umove([cube_params["X1"], 0., cube_params["Z1"]])[0], Umove([cube_params["X1"], 0., cube_params["Z1"]])[1], cube_params["Z1"]+Umove([cube_params["X1"], 0., cube_params["Z1"]])[2]], tol=1e-4)
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
            pinpoint_sd = dmech.PinpointSubDomain(coords=[0., cube_params["Y1"], cube_params["Z1"]], tol=1e-3)
        else :
            # print("mesh moved")
            pinpoint_sd = dmech.PinpointSubDomain(coords=[ Umove([0., cube_params["Y1"], cube_params["Z1"]])[0], cube_params["Y1"] + Umove([0., cube_params["Y1"], cube_params["Z1"]])[1], cube_params["Z1"]+Umove([0., cube_params["Y1"], cube_params["Z1"]])[2]], tol=1e-4)
        problem.add_constraint(
            V=problem.get_displacement_function_space().sub(2), 
            val=0.,
            sub_domain=pinpoint_sd,
            method='pointwise')
    

    Deltat = step_params.get("Deltat", 1.)
    dt_ini = step_params.get("dt_ini", 1.)
    dt_min = step_params.get("dt_min", 1.)
    dt_max = step_params.get("dt_max", 1.)
    k_step = problem.add_step(
        Deltat=Deltat,
        dt_ini=dt_ini,
        dt_min=dt_min,
        dt_max=dt_max)

    if multimaterial:
        rho_solid = mat_params[0].get("parameters").get("rho_solid", 1e-6)
    else:
        rho_solid = mat_params.get("parameters").get("rho_solid", 1e-6)

    load_type = load_params.get("type", "internal")

    problem.add_inertia_operator(
        measure=problem.dV,
        rho_val=float(inertia_val),
        k_step=k_step)

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
        f = load_params.get("f", 1e-3)
        problem.add_volume_force_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim,
            F_fin=[0.]*(dim-1) + [f],
            k_step=k_step)
    elif(load_type=="surface_pressure"):
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
        f = load_params.get("f", 1e4)
        P0 = load_params.get("P0", -0.5)
        Xm = load_params.get("Xm", mesh.coordinates()[:, 1].min() - mesh.coordinates()[:, 1].max())
        problem.add_surface_pressure_gradient0_loading_operator(
            dV=problem.dV,
            dS=problem.dS,
            f_ini=[0.]*dim,
            f_fin=[0., f, 0.],
            rho_solid=rho_solid,
            phis=problem.phis,
            P0_ini=0.,
            P0_fin=P0,
            # Xm=Xm,
            k_step=k_step)
    elif (load_type == "pgra"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        f = load_params.get("f", 1e4)
        P0 = load_params.get("P0", -0.5)
        Xm = load_params.get("Xm", mesh.coordinates()[:, 1].min() - mesh.coordinates()[:, 1].max())
        problem.add_surface_pressure_gradient_loading_operator(
            dV=problem.dV,
            dS=problem.dS,
            f_ini=[0.]*dim,
            f_fin=[0., f, 0.],
            rho_solid=rho_solid,
            P0_ini=0.,
            P0_fin=P0,
            # Xm=Xm,
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
        # relax_type="aitken",
        # relax_type="constant",
        relax_type = "backtracking",
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
    # print("mesh coordinates are", mesh.coordinates())

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
        # print(foi.name)
        if foi.name == "Phis0":
            phi = foi.func.vector().get_local()
            Phis0 = foi.func.vector().get_local()
            Phis0_func = foi.func
        if foi.name == "phis":
            phis = foi.func.vector().get_local()
            phis_func = foi.func
        if foi.name == "E":
            green_lagrange_tensor = foi.func
 
    phi0 = 0.
    for qoi in problem.qois:
        if qoi.name == "Phis0":
            phi0 = qoi.value
    
    # print("deformed mesh is", mesh.coordinates())
    # for i in range(0, len( mesh.coordinates())):
        # print(mesh_old[i]- mesh.coordinates()[i])

    print ("green-lagrange tensor", (dolfin.assemble(dolfin.inner(green_lagrange_tensor,green_lagrange_tensor)*V)/2/dolfin.assemble(dolfin.Constant(1)*V)))

    # print (problem.get_lbda0_subsol().func.vector().get_local())
    if not inverse:
        print(problem.get_x0_direct_subsol().func.vector().get_local())
        # print(problem.get_porosity_subsol().func.vector().get_local())
        
    # print("phi=", phi)

    # if inverse:

        # print("force equilibrium")
        # print("rhog", dolfin.assemble(rho_solid*phis_func*f*problem.dV))

        # p = dolfin.dot(problem.get_lbda_subsol().func.vector().get_local(), problem.mesh_normals) + dolfin.dot(problem.get_mu_subsol().func.vector().get_local(), dolfin.cross( problem.X + problem.get_x0_mass(), problem.mesh_normals) )
        # print("p=", p)
        
        # print("pnx equation", dolfin.assemble(-problem.get_p_subsol().func*problem.mesh_normals[0] * problem.dS ))
        # print("pny equation", dolfin.assemble(-problem.get_p_subsol().func*problem.mesh_normals[1] * problem.dS ))
        # print("pnz equation", dolfin.assemble(-problem.get_p_subsol().func*problem.mesh_normals[2] * problem.dS ))

        # print("momentum equilibrium")
        # print("momentum rhog x",  - dolfin.assemble( (problem.X[2]-problem.get_x0_mass()[2])* rho_solid*phis_func*f*problem.dV)) 
        # print("momentum rhog z",   dolfin.assemble( (problem.X[1]-problem.get_x0_mass()[0])* rho_solid*phis_func*f*problem.dV)) 

        # print("momentum x pn",   dolfin.assemble( ((problem.X[1]-problem.get_x0_mass()[1]) * problem.mesh_normals[2] - (problem.X[2]-problem.get_x0_mass()[2]) * problem.mesh_normals[1])* problem.get_p_subsol().func *problem.dS)) 
        # print("momentum y pn",   dolfin.assemble( ((problem.X[0]-problem.get_x0_mass()[0]) * problem.mesh_normals[2] - (problem.X[2]-problem.get_x0_mass()[2]) * problem.mesh_normals[0])* problem.get_p_subsol().func *problem.dS)) 
        # print("momentum z pn",   dolfin.assemble( ((problem.X[0]-problem.get_x0_mass()[0]) * problem.mesh_normals[1] - (problem.X[1]-problem.get_x0_mass()[1]) * problem.mesh_normals[0])* problem.get_p_subsol().func *problem.dS)) 

        # print("---> p = p0 + lmda.n + mu . x x n", dolfin.assemble(((problem.get_p_subsol().func - dolfin.Constant(P0) - dolfin.dot(dolfin.Constant(problem.get_lbda_subsol().func.vector().get_local()), problem.mesh_normals) - dolfin.dot(dolfin.Constant(problem.get_mu_subsol().func.vector().get_local()), dolfin.cross(problem.X - dolfin.Constant( problem.get_x0_mass()), problem.mesh_normals)))*(problem.get_p_subsol().func - dolfin.Constant(P0) - dolfin.dot(dolfin.Constant(problem.get_lbda_subsol().func.vector().get_local()), problem.mesh_normals) - dolfin.dot(dolfin.Constant(problem.get_mu_subsol().func.vector().get_local()), dolfin.cross(problem.X - dolfin.Constant( problem.get_x0_mass()), problem.mesh_normals))) )* problem.dS ) )

        # print("lbda.n",  dolfin.dot(dolfin.Constant(problem.get_lbda_subsol().func.vector().get_local()), problem.mesh_normals))

        # print("integral p0=", dolfin.assemble(dolfin.Constant(P0)*problem.dS))
        # print("integral lbda.n=", dolfin.assemble(dolfin.dot(dolfin.Constant(problem.get_lbda_subsol().func.vector().get_local()), problem.mesh_normals)*problem.dS))
        # print("integral  mu . x x n", dolfin.assemble(dolfin.dot(dolfin.Constant(problem.get_mu_subsol().func.vector().get_local()), dolfin.cross(problem.X - dolfin.Constant( problem.get_x0_mass()), problem.mesh_normals)) *problem.dS))
        # print("integral p=", dolfin.assemble(problem.get_p_subsol().func*problem.dS))


        # n = get_real_n_ext(mesh)
        # print("n_glob",  n)
        # Interpolate mesh values into boundary function

        # n = dolfin.FacetNormal(mesh)

        # Approximate facet normal in a suitable space using projection
        # V = dolfin.VectorFunctionSpace(mesh, "RT", 1)
        # u_ = dolfin.TrialFunction(V)
        # v_ = dolfin.TestFunction(V)
        # a = dolfin.inner(u_,v_)*dolfin.ds
        # l = dolfin.inner(n, v_)*dolfin.ds
        # A = dolfin.assemble(a, keep_diagonal=True)
        # L = dolfin.assemble(l)

        # A.ident_zeros()
        # nh = dolfin.Function(V)

        # dolfin.solve(A, nh.vector(), L)
        # dolfin.solve(A, nh.vector(), L)
        # dolfin.File("nh.pvd") << nh
    # else:
        # print("forces equilibrium")
        # print("rhog", dolfin.assemble(rho_solid * Phis0_func * f * problem.dV))
        # nf = dolfin.dot(problem.mesh_normals, dolfin.inv(problem.kinematics.F))
        # nf_norm = (1/2 * dolfin.inner(nf,nf))**(1/2)
        # n = nf/nf_norm
        # x = problem.X + problem.get_displacement_subsol().func
        # x_tilde = x-dolfin.Constant(problem.get_x0_direct_subsol().func.vector().get_local())
        # F = dolfin.Constant([0,f,0])

        # print("x0=", problem.get_x0_direct_subsol().func.vector().get_local() )
        # print("lbda=", problem.get_lbda_subsol().func.vector().get_local() )

        # print("mu=", problem.get_x0_direct_subsol().func.vector().get_local() )


        # print("integral of rho x", dolfin.assemble(phis_func * x[0] * problem.dV)/dolfin.assemble(phis_func * problem.dV))
        # print("integral of rho y", dolfin.assemble(phis_func * x[1] * problem.dV)/dolfin.assemble(phis_func * problem.dV))
        # print("integral of rho z", dolfin.assemble(phis_func * x[2] * problem.dV)/dolfin.assemble(phis_func * problem.dV))

        # print("pnx equation", dolfin.assemble(-problem.get_p_subsol().func*n[0] * problem.kinematics.J * nf_norm *problem.dS ))
        # print("pny equation", dolfin.assemble(-problem.get_p_subsol().func*n[1] * problem.kinematics.J * nf_norm *problem.dS ))
        # print("pnz equation", dolfin.assemble(-problem.get_p_subsol().func*n[2] * problem.kinematics.J * nf_norm *problem.dS ))

        # print("momentum equilibrium")
        # print("momentum rhog x",  dolfin.assemble( dolfin.cross(x_tilde,F)[0] * rho_solid * phis_func * problem.dV)) 
        # print("momentum rhog y",  dolfin.assemble( dolfin.cross(x_tilde,F)[1] * rho_solid * phis_func * problem.dV)) 
        # print("momentum rhog z",  dolfin.assemble( dolfin.cross(x_tilde,F)[2] * rho_solid * phis_func * problem.dV)) 

        # print("momentum pn x",   dolfin.assemble( (x_tilde[1] * n[2] - x_tilde[2] * n[1])* problem.get_p_subsol().func * problem.kinematics.J * nf_norm * problem.dS)) 
        # print("momentum pn y",   dolfin.assemble( (x_tilde[0] * n[2] - x_tilde[2] * n[0])* problem.get_p_subsol().func * problem.kinematics.J * nf_norm * problem.dS)) 
        # print("momentum pn z",   dolfin.assemble( (x_tilde[0] * n[1] - x_tilde[1] * n[0]) * problem.get_p_subsol().func * problem.kinematics.J * nf_norm * problem.dS)) 
        
        # print("p = p0 + lmda.n + mu . x x n", dolfin.assemble((problem.get_p_subsol().func - dolfin.Constant(P0) - dolfin.dot(dolfin.Constant(problem.get_lbda_subsol().func.vector().get_local()), n) - dolfin.dot(dolfin.Constant(problem.get_mu_subsol().func.vector().get_local()), dolfin.cross(x_tilde, n)) )* problem.kinematics.J * nf_norm* problem.dS ) )


        # P = dolfin.dot(dolfin.Constant(problem.get_lbda_subsol().func.vector().get_local()), n) + dolfin.dot(dolfin.Constant(problem.get_mu_subsol().func.vector().get_local()), dolfin.cross(x_tilde, n))
        # # print("P", P)
        # print("pn=", dolfin.assemble((P*n[0])*problem.kinematics.J * nf_norm *problem.dS))
        # print("pn=", dolfin.assemble((P*n[1])*problem.kinematics.J * nf_norm*problem.dS))
        # print("pn=", dolfin.assemble((P*n[2])*problem.kinematics.J * nf_norm*problem.dS))

    # print("lbda=", problem.get_lbda_subsol().func.vector().get_local())
    # print("mu=", problem.get_mu_subsol().func.vector().get_local())
    

    # print("phi0=", phi0)
    # print("phi average", dolfin.assemble(phi_test*V)/dolfin.assemble(1*V))
    # print(problem.get_displacement_subsol().func.vector()[:])
    get_J = False
    if get_results_fields:
        # phis0 = problem.get_porosity_subsol().func.vector().get_local()
        porosity_filename = res_basename+"-poro_final.xml"
        n_cells = len(mesh.cells())
        if inverse:
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <mesh_function type="double" dim="'+str(dim)+'" size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    file.write('    <entity index="'+str(k_cell)+'" value="'+str(Phis0[k_cell])+'"/>\n')
                file.write('  </mesh_function>\n')
                file.write('</dolfin>\n')
                file.close()
        else:
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <mesh_function type="double" dim="'+str(dim)+'" size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    file.write('    <entity index="'+str(k_cell)+'" value="'+str(phis[k_cell])+'"/>\n')
                file.write('  </mesh_function>\n')
                file.write('</dolfin>\n')
                file.close()
        if get_J:
            return(problem.get_displacement_subsol().func,  phi, V, problem.get_deformed_volume_subsol().func.vector()[0])
        else:
            return(problem.get_displacement_subsol().func,  phi, V)
    else:
        return
    

# def get_real_n_ext(mesh):

    
#     # mesh = UnitSquareMesh(10, 10)

#     ufc_element = UFCTetrahedron() if mesh.ufl_cell() == dolfin.tetrahedron else UFCTriangle()

#     tdim = mesh.topology().dim()
#     fdim = tdim - 1
#     top = mesh.topology()
#     mesh.init(tdim, fdim)
#     mesh.init(fdim, tdim)
#     f_to_c = top(fdim, tdim)

#     bndry_facets = []
#     for facet in dolfin.facets(mesh):
#         f_index = facet.index()
#         cells = f_to_c(f_index)
#         if len(cells) == 1:
#             bndry_facets.append(f_index)

#     # As we are using simplices, we can compute the Jacobian at an arbitrary point
#     fiat_el = Lagrange(ufc_simplex(tdim), 1)  
#     dphi_ = fiat_el.tabulate(1, numpy.array([0,]*tdim))
#     dphi = numpy.zeros((tdim, dolfin.Cell(mesh, 0).num_entities(0)), dtype=numpy.float64)
#     for i in range(tdim):
#         t = tuple()
#         for j in range(tdim):
#             if i == j:
#                 t += (1, )
#             else:
#                 t += (0, )
#         dphi[i] = dphi_[t]

#     test = []
#     lbda = [0.,1.,0.]

    # c_to_f = top(tdim, fdim)
    # for facet in bndry_facets:
    #     cell = f_to_c(facet)[0]
    #     facets = c_to_f(cell)
    #     local_index = numpy.flatnonzero(facets==facet)[0]
    #     normal = ufc_element.compute_reference_normal(fdim, local_index)
    #     coord_dofs = mesh.cells()[cell]
    #     coords = mesh.coordinates()[coord_dofs]
    #     J = numpy.dot(dphi, coords).T
    #     Jinv = numpy.linalg.inv(J)
    #     print("J=", J)
    #     n_glob = numpy.dot(Jinv.T, normal)
    #     n_glob = n_glob/numpy.linalg.norm(n_glob)
    #     test.append(dolfin.dot(dolfin.Constant(lbda), dolfin.Constant(n_glob)))
    # return(n_glob)
