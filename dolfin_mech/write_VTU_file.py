#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import os
import shutil

import dolfin_mech as dmech

################################################################################

def write_VTU_file(
        filebasename,
        function,
        time,
        zfill=3):

    file_pvd = dolfin.File(filebasename+"__.pvd")
    file_pvd << (function, float(time))
    os.remove(
        filebasename+"__.pvd")
    shutil.move(
        filebasename+"__"+"".zfill(6)+".vtu",
        filebasename+"_"+str(time).zfill(zfill)+".vtu")
