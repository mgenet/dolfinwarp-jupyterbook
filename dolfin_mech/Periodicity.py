#################################################################
###                                                           ###
### Created by Felipe Álvarez, 2019-2021                      ###
###                                                           ###
### Pontificia Universidad Católica de Chile, Santiago, Chile ###
###                                                           ###
#################################################################

#coding=utf8

import dolfin
import numpy as np


class Periodic_Boundary(dolfin.SubDomain):
    def __init__(self, vertices, tolerance):
        """ vertices stores the coordinates of the 4 unit cell corners"""
        dolfin.SubDomain.__init__(self, tolerance)
        self.tol = tolerance
        self.vv = vertices
        self.a1 = self.vv[1,:]-self.vv[0,:] # first vector generating periodicity
        self.a2 = self.vv[3,:]-self.vv[0,:] # second vector generating periodicity
        # check if UC vertices form indeed a parallelogram
        assert np.linalg.norm(self.vv[2, :]-self.vv[3, :] - self.a1) <= self.tol
        assert np.linalg.norm(self.vv[2, :]-self.vv[1, :] - self.a2) <= self.tol

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the
        # bottom-right or top-left vertices
        return bool((dolfin.near(x[0], self.vv[0,0] + x[1]*self.a2[0]/self.vv[3,1], self.tol) or
                     dolfin.near(x[1], self.vv[0,1] + x[0]*self.a1[1]/self.vv[1,0], self.tol)) and
                     (not ((dolfin.near(x[0], self.vv[1,0], self.tol) and dolfin.near(x[1], self.vv[1,1], self.tol)) or
                     (dolfin.near(x[0], self.vv[3,0], self.tol) and dolfin.near(x[1], self.vv[3,1], self.tol)))) and on_boundary)

    def map(self, x, y):
        if dolfin.near(x[0], self.vv[2,0], self.tol) and dolfin.near(x[1], self.vv[2,1], self.tol): # if on top-right corner
            y[0] = x[0] - (self.a1[0]+self.a2[0])
            y[1] = x[1] - (self.a1[1]+self.a2[1])
        elif dolfin.near(x[0], self.vv[1,0] + x[1]*self.a2[0]/self.vv[2,1], self.tol): # if on right boundary
            y[0] = x[0] - self.a1[0]
            y[1] = x[1] - self.a1[1]
        else:   # should be on top boundary
            y[0] = x[0] - self.a2[0]
            y[1] = x[1] - self.a2[1]
