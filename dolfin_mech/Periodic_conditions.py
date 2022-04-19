#################################################################
###                                                           ###
### Created by Felipe Álvarez, 2019-2021                      ###
###                                                           ###
### Pontificia Universidad Católica de Chile, Santiago, Chile ###
###                                                           ###
#################################################################

#coding=utf8

import dolfin
import dolfin_mech as dmech


def Periodic_all_faces(x_min, x_max, y_min, y_max, tol=dolfin.DOLFIN_EPS):
    """Create periodic boundary conditions in all faces"""

    class PeriodicBoundary(dolfin.SubDomain):

        # def __init__(self,
        #             dx_length,
        #             dy_length):
        #     self.dx_length = dx_length
        #     self.dy_length = dy_length
        #     # self.tol = tol
        #     tol=dolfin.DOLFIN_EPS

        dx_length = x_max - x_min
        dy_length = y_max - y_min

        def inside(self, x, on_boundary):
            return bool(on_boundary and (dolfin.near(x[0],x_min,eps=tol) or dolfin.near(x[1],y_min,eps=tol) and not (dolfin.near(x[0],x_max,eps=tol) or dolfin.near(x[1],y_max,eps=tol))) )

        # Take as reference vertex (dx_length, dy_length, dz_length)
        def map(self, x, y):

            # Vertex

            if dolfin.near(x[0], x_max, eps=tol) and dolfin.near(x[1], y_max, eps=tol):
                y[0] = x[0] - dx_length
                y[1] = x[1] - dy_length


            elif dolfin.near(x[0], x_max, eps=tol) and dolfin.near(x[1], y_min, eps=tol):
                y[0] = x[0] - dx_length
                y[1] = x[1]

            elif dolfin.near(x[0], x_min, eps=tol) and dolfin.near(x[1], y_max, eps=tol):
                y[0] = x[0]
                y[1] = x[1] - dy_length


            elif dolfin.near(x[0], x_min, eps=tol) and dolfin.near(x[1], y_min, eps=tol):
                y[0] = x[0]
                y[1] = x[1]


            # Edges

            elif dolfin.near(x[0], x_min, eps=tol) and dolfin.near(x[1],y_max, eps=tol):
                y[0] = x[0]
                y[1] = x[1] - dy_length

            elif dolfin.near(x[0], x_max, eps=tol) and dolfin.near(x[1],y_max, eps=tol):
                y[0] = x[0] - dx_length
                y[1] = x[1] - dy_length



            else:
                y[0] = -1000.
                y[1] = -1000.

            # Else clause is there to have the point always end up mapped somewhere
            # https://fenicsproject.discourse.group/t/periodic-boundary-class-for-2-target-domains/99/2

            # Check vertices
            # Check if is necessary to define a spacial case
            # https://fenicsproject.org/qa/262/possible-specify-more-than-one-periodic-boundary-condition/

            # Mesh must be periodic
            # https://fenicsproject.org/qa/5812/periodic-boundary-condition-for-meshes-that-are-not-built-in/

    return PeriodicBoundary()
