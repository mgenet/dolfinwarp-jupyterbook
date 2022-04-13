#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech

################################################################################

class QOI():



    def __init__(self,
            name,
            expr,
            norm=1.,
            form_compiler_parameters={}):

        self.name = name
        self.expr = expr
        self.norm = norm
        self.form_compiler_parameters = form_compiler_parameters



    def update(self):

        # print(self.name)
        # print(self.expr)
        # print(self.form_compiler_parameters)

        self.value = dolfin.assemble(
            self.expr,
            form_compiler_parameters=self.form_compiler_parameters)
        self.value /= self.norm
