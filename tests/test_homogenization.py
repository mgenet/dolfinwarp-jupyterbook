#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

#################################################################### imports ###

import sys
import gmsh
import meshio
import dolfin
import numpy as np

import myPythonLibrary as mypy
import dolfin_mech     as dmech

####################################################################### test ###
def homogenization(
    dim,
    mat_params,
    res_basename):

    ################################################################### Mesh ###

    xmin = 0.
    ymin = 0.
    zmin = 0.
    xmax = 1.
    ymax = 1.
    zmax = 1.
    x0 = 0.3
    y0 = 0.3
    z0 = 0.3
    r0 = 0.2
    l = 0.10
    e = 1e-6

    def setPeriodic(coord):
        # From https://gitlab.onelab.info/gmsh/gmsh/-/issues/744
        smin = gmsh.model.getEntitiesInBoundingBox(xmin - e, ymin - e, zmin - e,
                                                    (xmin + e) if (coord == 0) else (xmax + e),
                                                    (ymin + e) if (coord == 1) else (ymax + e),
                                                    (zmin + e) if (coord == 2) else (zmax + e),
                                                    2)
        dx = (xmax - xmin) if (coord == 0) else 0
        dy = (ymax - ymin) if (coord == 1) else 0
        dz = (zmax - zmin) if (coord == 2) else 0

        for i in smin:
            bb = gmsh.model.getBoundingBox(i[0], i[1])
            bbe = [bb[0] - e + dx, bb[1] - e + dy, bb[2] - e + dz,
                    bb[3] + e + dx, bb[4] + e + dy, bb[5] + e + dz]
            smax = gmsh.model.getEntitiesInBoundingBox(bbe[0], bbe[1], bbe[2],
                                                        bbe[3], bbe[4], bbe[5])
            for j in smax:
                bb2 = list(gmsh.model.getBoundingBox(j[0], j[1]))
                bb2[0] -= dx; bb2[1] -= dy; bb2[2] -= dz;
                bb2[3] -= dx; bb2[4] -= dy; bb2[5] -= dz;
                if ((abs(bb2[0] - bb[0]) < e) and (abs(bb2[1] - bb[1]) < e) and
                    (abs(bb2[2] - bb[2]) < e) and (abs(bb2[3] - bb[3]) < e) and
                    (abs(bb2[4] - bb[4]) < e) and (abs(bb2[5] - bb[5]) < e)):
                    gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], [1, 0, 0, dx,\
                                                                    0, 1, 0, dy,\
                                                                    0, 0, 1, dz,\
                                                                    0, 0, 0, 1 ])

    gmsh.initialize()
    box_tag = 1
    hole_tag = 2
    rve_tag = 3
    if (dim==2):
        gmsh.model.occ.addRectangle(x=xmin, y=ymin, z=0, dx=xmax-xmin, dy=ymax-ymin, tag=box_tag)
        gmsh.model.occ.addDisk(xc=x0, yc=y0, zc=0, rx=r0, ry=r0, tag=hole_tag)
        gmsh.model.occ.cut(objectDimTags=[(2, box_tag)], toolDimTags=[(2, hole_tag)], tag=rve_tag)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(dim=2, tags=[rve_tag])
        gmsh.model.mesh.setPeriodic(dim=1, tags=[2], tagsMaster=[1], affineTransform=[1, 0, 0, xmax-xmin,\
                                                                                        0, 1, 0, 0        ,\
                                                                                        0, 0, 1, 0        ,\
                                                                                        0, 0, 0, 1        ])
        gmsh.model.mesh.setPeriodic(dim=1, tags=[4], tagsMaster=[3], affineTransform=[1, 0, 0, 0        ,\
                                                                                        0, 1, 0, ymax-ymin,\
                                                                                        0, 0, 1, 0        ,\
                                                                                        0, 0, 0, 1        ])
        gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=l)
        gmsh.model.mesh.generate(dim=2)
    elif (dim==3):
        gmsh.model.occ.addBox(x=xmin, y=ymin, z=zmin, dx=xmax-xmin, dy=ymax-ymin, dz=zmax-zmin, tag=box_tag)
        gmsh.model.occ.addSphere(xc=x0, yc=y0, zc=z0, radius=r0, tag=hole_tag)
        gmsh.model.occ.cut(objectDimTags=[(3, box_tag)], toolDimTags=[(3, hole_tag)], tag=rve_tag)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(dim=3, tags=[rve_tag])
        setPeriodic(0)
        setPeriodic(1)
        setPeriodic(2)
        # gmsh.model.mesh.setPeriodic(dim=2, tags=[2], tagsMaster=[1], affineTransform=[1, 0, 0, xmax-xmin,\
        #                                                                               0, 1, 0, 0        ,\
        #                                                                               0, 0, 1, 0        ,\
        #                                                                               0, 0, 0, 1        ])
        # gmsh.model.mesh.setPeriodic(dim=2, tags=[4], tagsMaster=[3], affineTransform=[1, 0, 0, 0        ,\
        #                                                                               0, 1, 0, ymax-ymin,\
        #                                                                               0, 0, 1, 0        ,\
        #                                                                               0, 0, 0, 1        ])
        # gmsh.model.mesh.setPeriodic(dim=2, tags=[6], tagsMaster=[5], affineTransform=[1, 0, 0, 0        ,\
        #                                                                               0, 1, 0, 0        ,\
        #                                                                               0, 0, 1, zmax-zmin,\
        #                                                                               0, 0, 0, 1        ])
        gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=l)
        gmsh.model.mesh.generate(dim=3)
    gmsh.write(res_basename+"-mesh.vtk")
    gmsh.finalize()

    mesh = meshio.read(res_basename+"-mesh.vtk")
    if (dim==2): mesh.points = mesh.points[:, :2]
    meshio.write(res_basename+"-mesh.xdmf", mesh)

    mesh = dolfin.Mesh()
    dolfin.XDMFFile(res_basename+"-mesh.xdmf").read(mesh)

    # if   (dim==2):
    #     geometry = mshr.Rectangle(dolfin.Point(xmin, ymin), dolfin.Point(xmax, xmax))\
    #              - mshr.Circle(dolfin.Point(x0, y0), r0)
    # elif (dim==3):
    #     geometry = mshr.Box(dolfin.Point(xmin, ymin, zmin), dolfin.Point(xmax, ymax, zmax))\
    #              - mshr.Sphere(dolfin.Point(x0, y0, z0), r0)
    # mesh = mshr.generate_mesh(geometry, 10)

    ################################################## Homogenization ###
    vertices = np.array([[xmin, ymin],
                         [xmax, ymin],
                         [xmax, ymax],
                         [xmin, ymax]])

    homog = dmech.HomogenizedParameters(
                dim=dim,
                mesh=mesh,
                mat_params=mat_params,
                vertices=vertices,
                vol=1,
                bbox=[0., 1.]*dim)

    homogenized_parameters = homog.homogenized_param()

    qoi_printer = mypy.DataPrinter(
            names=["E_hom", "nu_hom"],
            filename=res_basename+"-qois.dat",
            limited_precision=False)

    qoi_printer.write_line([mat_params["E"], mat_params["nu"]])
    qoi_printer.write_line([homogenized_parameters[0], homogenized_parameters[1]])


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

    print("dim =",dim)

    res_basename  = sys.argv[0][:-3]
    res_basename += "-dim="+str(dim)

    homogenization(
        dim=dim,
        mat_params={"E":1.0, "nu":0.3},
        res_basename=res_folder+"/"+res_basename)

    test.test(res_basename)

