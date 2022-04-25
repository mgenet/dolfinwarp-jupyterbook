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
            form_compiler_parameters={},
            point=None,
            update_type="assembly"):

        self.name                     = name
        self.expr                     = expr
        self.norm                     = norm
        self.form_compiler_parameters = form_compiler_parameters
        self.point                    = point

        if (update_type == "assembly"):
            self.update = self.update_assembly
        elif (update_type == "direct"):
            self.update = self.update_direct



    def update_assembly(self):

        # print(self.name)
        # print(self.expr)
        # print(self.form_compiler_parameters)

        self.value = dolfin.assemble(
            self.expr,
            form_compiler_parameters=self.form_compiler_parameters)
        self.value /= self.norm


    def update_direct(self):
        
        self.value = self.expr(self.point)
        self.value /= self.norm
