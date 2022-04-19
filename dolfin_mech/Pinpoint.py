#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2020                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy as np
import dolfin_mech as dmech

################################################################################

class Pinpoint(dolfin.SubDomain):
    
    def __init__(self, coords, tol=None):
        """Apply Dirichlet BCs at one point"""
        """Takes a coords argument (of type tuple, list or np.array) allowing specification of coordinates where you want to apply BC"""
        self.coords = np.array(coords)
        self.tol = tol if (tol is not None) else 1e-5
        dolfin.SubDomain.__init__(self)


    def move(self, coords):
        """Use to change the point without a need of creating new DirichletBC instance - especially when mesh moves"""
        self.coords[:] = np.array(coords)


    def inside(self, x, on_boundary):
        return np.linalg.norm(x-self.coords) < self.tol


    def check_inside(self, mesh):
        """Check if there are nodes within Pinpoint.inside()"""
        x_coords = []
        for x in mesh.coordinates():
            if self.inside(x, True):
                x_coords.append(x)
        return np.reshape(x_coords, (-1,2))
