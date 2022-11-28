

import dolfin
import dolfin_mech     as dmech
import argparse
import numpy
import copy

import sys



################################################################################

def test() :

    Pmax_inspi, Pmin_inspi = -0.8, -0.3
    Pmax_expi, Pmin_expi = -0.6, -0.1

########## material parameters the units are T mm s, hence MPa
    mat_params = {
        "alpha":0.16,
        "gamma":0.5,
        "c1":0.2,
        "c2":0.4,
        "kappa":1e2,
        "eta":1e-3}

####### inverse problem

#### loads inverse problem: gravity + pressure; the units are T mm s

    # load_params_inverse = {
    #         "type":"pgra0", "f":9.81e-6*0.5, "DP": 9.81e-6*0.5, "P0" : 0, "X0":[0.,0.,100.]}


###### inverse problem

    # Uref, U1, Phis1, phi1, V = dmech.RivlinCube_PoroHyperelasticity(
    #                     dim=3,
    #                     inverse=1,
    #                     porosity_params={"type": "mesh_function_xml", "val":0.5},
    #                     cube_params={"X1":100, "Y1":100, "Z1":100, "l":50},
    #                     mat_params={"scaling":"linear", "parameters":mat_params},
    #                     step_params={"dt_min":0.1},
    #                     load_params=load_params_direct,
    #                     res_basename = "/Users/peyrault/Documents/test",
    #                     plot_curves=1,
    #                     verbose=1,
    #                     test = False,
    #                     BC = 1)






########################################################################################### test inverse problem: gravity + pressure gradient + BC

###### BC1 "reference"

    # Uref, Rx1, Ry1, Rz1, U1, Phis1, phi1, V = dmech.RivlinCube_PoroHyperelasticity(
    #                     dim=3,
    #                     inverse=1,
    #                     porosity_params={"type": "mesh_function_xml", "val":0.5},
    #                     cube_params={"X1":100, "Y1":100, "Z1":100, "l":10},
    #                     mat_params={"scaling":"linear", "parameters":mat_params},
    #                     step_params={"dt_min":0.1},
    #                     load_params=load_params_inverse,
    #                     res_basename = "/Users/peyrault/Documents/test1",
    #                     plot_curves=1,
    #                     verbose=1,
    #                     test = True,
    #                     BC = 1)

###### BC2 "comparison"

    # Utest, Rx2, Ry2, Rz2, U2, Phis2, phi2, V2 = dmech.RivlinCube_PoroHyperelasticity(
    #                     dim=3,
    #                     inverse=1,
    #                     porosity_params={"type": "mesh_function_xml", "val":0.5},
    #                     cube_params={"X1":100, "Y1":100, "Z1":100, "l":10},
    #                     mat_params={"scaling":"linear", "parameters":mat_params},
    #                     step_params={"dt_min":0.1},
    #                     load_params=load_params_inverse,
    #                     res_basename = "/Users/peyrault/Documents/test2",
    #                     plot_curves=1,
    #                     verbose=1,
    #                     BC = 2)


##### rotation + displacements for rigid body motion
    
    # Udiff= copy.deepcopy(Utest)

    ### rigid body displacement
    # ux, uy, uz =  Udiff[0][0],  Udiff[0][1],  Udiff[0][2]

#### for max difference between reference and comparison
    # max_diff = 0.
    # first = 0.
    # second = 0.

    # for i in range(0, len(Udiff)):
    #     Udiff[i][0] -= ux
    #     Udiff[i][1] -= uy
    #     Udiff[i][2] -= uz
    #     print("ref",numpy.dot(Rx2, numpy.dot(Ry2, numpy.dot(Rz2,Uref[i][:]))))
    #     print("test", Udiff[i][:])
    #     test = numpy.dot(Rx2, numpy.dot(Ry2, numpy.dot(Rz2,Uref[i][:]))) - Udiff[i][:]
    #     max_new = max(abs(test[0]), abs(test[1]), abs(test[2]))
    #     if max_new > max_diff :
    #         max_diff = max_new
    #         first = numpy.dot(Rx2, numpy.dot(Ry2, numpy.dot(Rz2,Uref[i][:])))
    #         second = Udiff[i][:]

    # print(max_diff, first, second)


############################## 2nd option: max relative difference of position of the nodes
    
#     max = 0.
#     for i in range(0, len(Utest)):
#         test1, test2 = 0., 0. 
#         test1 = ( (Uref[0][0]-Uref[i][0])**2 + (Uref[0][1]-Uref[i][1])**2+ (Uref[0][2]-Uref[i][2])**2)**(1/2)
#         test2 = ((Utest[0][0]-Utest[i][0])**2+(Utest[0][1]-Utest[i][1])**2+(Utest[0][2]-Utest[i][2])**2)**(1/2)
#         if abs(test1-test2) > 0:
#             max = test1-test2
#     print(max)











################################################################################### reference --> inspi (direct problem) with the porosity computed for the direct problem + ALE.move(mesh)


####### parameters for the direct problem
    # load_params_direct = {}


    # load_params_direct = {
    #     "type":"pgra", "f":9.81e-6*0.5, "DP":9.81e-6*0.5, "P0" : 0, "X0":[100.,100.,100.]}

    # position, U2, Phis2, phi2, V2 = dmech.RivlinCube_PoroHyperelasticity(
    #                     dim=3,
    #                     inverse=0,
    #                     porosity_params={"type": "mesh_function_xml", "val":0.5},
    #              #       porosity_params= {"type": "mesh_function_xml_custom", "val":Phis1},  ##### getting porosity field from previous caclulation
    #                     cube_params={"X1":100, "Y1":100, "Z1":100, "l":50},
    #                     mat_params={"scaling":"linear", "parameters":mat_params},
    #                     step_params={"dt_min":0.1},
    #                     load_params=load_params_direct,
    #                     res_basename = "/Users/peyrault/Documents/test_inspi",
    #                     # move = {"move":True, "U":U1},
    #                     plot_curves=0,
    #                     verbose=1, 
    #                     BC=1)

    # # print(U2.vector().get_local())
    
    # # # U = U1.copy(deepcopy=True)
    
    # # U.vector().set_local(U1.vector().get_local()[:]+U2.vector().get_local()[:])

    # return




########################################################################################### test direct problem: gravity + pressure gradient + BC

    # load_params_inverse = {
    #         "type":"pgra0", "f":-9.81e-3*0.5, "DP": -9.81e-3*0.5, "P0" : (Pmax_expi+ Pmin_expi)/2, "X0":[100.,100.,50.]}

    # load_params_direct = {
    #     "type":"pgra", "f":-9.81e-3*0.5, "F0":-9.81e-3*0.5, "P0" : (Pmax_inspi+ Pmin_inspi)/2, "X0":[50,50,50]}

    # # field1, Uref, Rx1, Ry1, Rz1, U1, Phis1, phi1, V1 = dmech.RivlinCube_PoroHyperelasticity(
    # Uref, Phis1, phi1, V1 = dmech.RivlinCube_PoroHyperelasticity(
    #                     dim=3,
    #                     inverse=0,
    #                     porosity_params={"type": "mesh_function_xml", "val":0.5},
    #                     # porosity_params= {"type": "mesh_function_xml_custom", "val":Phis1},  ##### getting porosity field from previous caclulation
    #                     cube_params={"X1":100, "Y1":100, "Z1":100, "l":50},
    #                     mat_params={"scaling":"linear", "parameters":mat_params},
    #                     step_params={"dt_min":0.1},
    #                     load_params=load_params_direct,
    #                     res_basename = "/Users/peyrault/Documents/test_inspi1",
    #                     # move = {"move":True, "U":U1},
    #                     plot_curves=0,
    #                     verbose=1, 
    #                     test = True,
    #                     BC=1)


    # # field2, Utest, Rx2, Ry2, Rz2, U2, Phis2, phi2, V2 = dmech.RivlinCube_PoroHyperelasticity(
    # Utest, Phis2, phi2, V2 = dmech.RivlinCube_PoroHyperelasticity(
    #                     dim=3,
    #                     inverse=0,
    #                     porosity_params={"type": "mesh_function_xml", "val":0.5},
    #              #       porosity_params= {"type": "mesh_function_xml_custom", "val":Phis1},  ##### getting porosity field from previous caclulation
    #                     cube_params={"X1":100, "Y1":100, "Z1":100, "l":50},
    #                     mat_params={"scaling":"linear", "parameters":mat_params},
    #                     step_params={"dt_min":0.1},
    #                     load_params=load_params_direct,
    #                     res_basename = "/Users/peyrault/Documents/test_inspi2",
    #                     # move = {"move":True, "U":U1},
    #                     plot_curves=0,
    #                     verbose=1, 
    #                     test = True,
    #                     BC=2)

    # test_rigidbodymotion = dmech.RigidBodyMotion(Uref=Uref, Utest=Utest, X0=0., Y0=0., Z0=0., X1=100., Y1=100., Z1=100., tol=1e-3)


    ### rotation + displacements for rigid body motion
    
#     Udiff= copy.deepcopy(Utest)

#     ## rigid body displacement
#     ux, uy, uz =  Udiff[0][0],  Udiff[0][1],  Udiff[0][2]

# ### for max difference between reference and comparison
#     max_diff = 0.
#     first = 0.
#     second = 0.

#     for i in range(0, len(Udiff)):
#         Udiff[i][0] -= ux
#         Udiff[i][1] -= uy
#         Udiff[i][2] -= uz
#         print("ref", Uref[i][:])
#         print("ref", numpy.dot(Rz2,Uref[i][:]))
#         print("ref",numpy.dot(Rx2, numpy.dot(Ry2, numpy.dot(Rz2,Uref[i][:]))))
#         print("test", Udiff[i][:])
#         test = numpy.dot(Rx2, numpy.dot(Ry2, numpy.dot(Rz2,Uref[i][:]))) - Udiff[i][:]
#         max_new = max(abs(test[0]), abs(test[1]), abs(test[2]))
#         if max_new > max_diff :
#             max_diff = max_new
#             first = numpy.dot(Rx2, numpy.dot(Ry2, numpy.dot(Rz2,Uref[i][:])))
#             second = Udiff[i][:]

#     print(max_diff, first, second)

    # print(field2(0.,0.,0.)) 
    # print(field2.vec())

    # for i in range (len(field))    





###################################################################################################### expi -> ref, ref -> inspi

    
    cube_params = {"X1":100., "Y1":100., "Z1":100., "l":50}

    load_params_inverse = {
        "type":"pgra0", "f":-9.81e-3*0.5, "DP": -9.81e-3*0.5, "P0" : (Pmax_expi+Pmin_expi)/2, "X0":[100.,100.,50.]}

    # load_params_direct = {
    #     "type":"pgra", "f":-9.81e-3*0.5, "F0":-9.81e-3*0.5, "P0" : (Pmax_inspi+Pmin_inspi)/2, "X0":[50.,50.,50.]}

    # print((Pmax_expi+ Pmin_expi)/2)

    U1, Phis1, phi1, V = dmech.RivlinCube_PoroHyperelasticity(
        dim=3,
        inverse=1,
        # porosity_params={"type": "constant", "val":0.5},
        porosity_params={"type": "mesh_function_xml", "val":0.5},
        cube_params=cube_params,
        mat_params={"scaling":"linear", "parameters":mat_params},
        step_params={"dt_min":0.1},
        load_params=load_params_inverse,
        res_basename = "/Users/peyrault/Documents/test_expi",
        plot_curves=1,
        verbose=1,
        BC = 1)

    # print(U1.vector().get_local())

    # print( (Pmax_inspi+ Pmin_inspi)/2)

    # load_params_direct = {
    #     "type":"pgra", "f":-9.81e-3*phi1, "F0":-9.81e-3*phi1, "P0" : (Pmax_inspi+ Pmin_inspi)/2, "X0":[100.,100.,50.]}

    # U2, Phis2, phi2, V2 = dmech.RivlinCube_PoroHyperelasticity(
    #                     dim=3,
    #                     inverse=0,
    #                     # porosity_params={"type": "mesh_function_xml", "val":0.5},
    #                     porosity_params= {"type": "mesh_function_xml_custom", "val":Phis1},  ##### getting porosity field from previous caclulation
    #                     cube_params=cube_params,
    #                     mat_params={"scaling":"linear", "parameters":mat_params},
    #                     step_params={"dt_min":1e-4},
    #                     load_params=load_params_direct,
    #                     res_basename = "/Users/peyrault/Documents/test_inspi",
    #                     move = {"move":True, "U":U1},
    #                     plot_curves=0,
    #                     verbose=1, 
    #                     BC=1)





if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    test()
