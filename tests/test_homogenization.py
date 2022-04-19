#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

#################################################################### imports ###


import dolfin_mech     as dmech
import gmsh
import sys

################################################################################

# gmsh.initialize()
# # gmsh.clear()
# model = gmsh.model
# occ = model.occ
# # mesh = model.mesh
# # model.add("myGeo")
# # gmsh.initialize(sys.argv)
# # lcar = 0.02

     
# # p1   = occ.addPoint(0, 0, 0, lcar)
# # p2   = occ.addPoint(1, 0, 0, lcar)
# # p3   = occ.addPoint(1, 1, 0, lcar)
# # p4   = occ.addPoint(0, 1, 0, lcar)

# # occ.addLine(p1, p2, 1)
# # occ.addLine(p2, p3, 2)
# # occ.addLine(p3, p4, 3)
# # occ.addLine(p4, p1, 4)

# # occ.addPlaneSurface([occ.addCurveLoop([1, 2, 3, 4])], 100)

# # circle = occ.addCircle(0,0,0,1,200)
# r=occ.addRectangle(0,0,0, 1,1)
# c=occ.addCircle(0.5,0.5,0, 0.2)
# w2=gmsh.model.occ.addWire([c])
# # occ.cut([(2, r)],[(2, c)])

# occ.synchronize()
# # # mat = gmsh.model.addPhysicalGroup(1, [100])
# # # mat = gmsh.model.addPhysicalGroup(2, [200])
# gmsh.model.mesh.generate()
# gmsh.write("tester.msh")

# # HP = dmech.HomogenizedParameters('2D_Periodic', 10, 0.2)


# # gmsh.model.occ.synchronize()

# # if '-nopopup' not in sys.argv:
# #     gmsh.fltk.run()

# # gmsh.finalize()

HP = dmech.HomogenizedParameters('2D_Periodic', 5, 0.2)
print(HP.homogenized_param())