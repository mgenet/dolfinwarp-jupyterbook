#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2020                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# from builtins import *

import dolfin
import time

import dolfin_mech as dmech

################################################################################

class FOI():



    def __init__(self,
            expr=None,
            fs=None,
            func=None,
            name=None,
            form_compiler_parameters={},
            update_type="local_solver"): # local_solver or project

        if (expr is not None) and (fs is not None):

            self.expr = expr
            self.fs = fs
            self.func = func if func is not None else dolfin.Function(fs)
            if (name is not None):
                self.name = name
                self.func.rename(self.name, self.name)

            if (update_type == "local_solver"):

                self.func_test = dolfin.TestFunction(self.fs)
                self.func_tria = dolfin.TrialFunction(self.fs)
                self.form_compiler_parameters = form_compiler_parameters

                self.a_expr = dolfin.inner(self.func_tria,
                                           self.func_test) * dolfin.dx(metadata=self.form_compiler_parameters)
                self.b_expr = dolfin.inner(self.expr,
                                           self.func_test) * dolfin.dx(metadata=self.form_compiler_parameters)
                self.local_solver = dolfin.LocalSolver(
                    self.a_expr,
                    self.b_expr)
                # t = time.time()
                self.local_solver.factorize()
                # t = time.time() - t
                # print("LocalSolver factorization = "+str(t)+" s")

                self.update = self.update_local_solver

            elif (update_type == "project"):

                self.update = self.update_project

        elif (expr is None) and (fs is None) and (func is not None):

            self.func = func

            self.update = self.update_none



    def update_local_solver(self):

        # print(self.name)
        # print(self.form_compiler_parameters)

        # t = time.time()
        self.local_solver.solve_local_rhs(self.func)
        # t = time.time() - t
        # print("LocalSolver solve = "+str(t)+" s")



    def update_project(self):

        # print(self.name)
        # print(self.form_compiler_parameters)

        # t = time.time()
        dolfin.project(
            v=self.expr,
            V=self.fs,
            function=self.func,
            form_compiler_parameters=self.form_compiler_parameters)
        # t = time.time() - t
        # print("Projec = "+str(t)+" s")



    def update_none(self):

        pass
