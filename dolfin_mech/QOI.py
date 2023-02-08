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
            constant=0.,
            divide_by_dt=False,
            form_compiler_parameters={},
            point=None,
            update_type="assembly"):

        self.name                     = name
        self.expr                     = expr
        self.norm                     = norm
        self.constant                 = constant
        self.divide_by_dt             = divide_by_dt
        self.form_compiler_parameters = form_compiler_parameters
        self.point                    = point

        if (update_type == "assembly"):
            self.update = self.update_assembly
        elif (update_type == "direct"):
            self.update = self.update_direct
        elif (update_type == "scalar"):
            self.update = self.update_scalar
        elif (update_type == "volume"):
            self.update = self.update_volume
        elif (update_type == "surface"):
            self.update = self.update_internal_surface



    def update_assembly(self, dt=None, t_step=None, kinematics=None, dV=None, dS=None):

        # print(self.name)
        # print(self.expr)
        # print(self.form_compiler_parameters)

        self.value = dolfin.assemble(
            self.expr,
            form_compiler_parameters=self.form_compiler_parameters)

        self.value += self.constant
        self.value /= self.norm

        if (self.divide_by_dt):
            assert (dt != 0),\
                "dt (="+str(dt)+") should be non zero. Aborting."
            self.value /= dt


        



    def update_direct(self, dt=None, t_step=None, kinematics=None, dV=None, dS=None):
        
        self.value = self.expr(self.point)

        self.value += self.constant
        self.value /= self.norm

        if (self.divide_by_dt) and (dt is not None):
            self.value /= dt

        # print("t_step = " +str(t_step))




    def update_scalar(self, dt=None, t_step=None, kinematics=None, dV=None, dS=None):
        
        self.value = self.expr 

        self.value += self.constant
        self.value /= self.norm

        # self.value *= t_step

        if (self.divide_by_dt) and (dt is not None):
            self.value /= dt

        # print("t_step = " +str(dt * self.value))



    def update_volume(self, dt=None, t_step=None, kinematics=None, dV=None, dS=None):
        
        self.value = self.expr 

        
        self.value += self.constant
        self.value /= self.norm

        self.value *= dolfin.assemble(kinematics.C[0,0]*dV)/dolfin.assemble(dolfin.Constant(1) * dV)



    def update_internal_surface(self, dt=None, t_step=None, kinematics=None, dV=None, dS=None):
        
        self.value = self.expr 

        
        self.value += self.constant
        self.value /= self.norm

        # self.value *= dolfin.assemble(dolfin.Constant(1))
        self.value *= dolfin.assemble(dolfin.Constant(1) * dS(0))


        