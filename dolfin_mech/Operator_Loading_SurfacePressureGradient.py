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
from .Operator import Operator

################################################################################



class SurfacePressureGradientLoadingOperator(Operator):

    def __init__(self,
            X,
            x0,
            x0_test,
            kinematics,
            U,
            U_test,
            Phis0,
            V0, 
            N,
            lbda,
            lbda_test,
            mu,
            mu_test,
            dS,
            dV,
            P0_val=None, P0_ini=None, P0_fin=None,
            f_val=None, f_ini=None, f_fin=None):

        print("surface gradient")
        print(f_val, f_ini, f_fin)

        self.V0 = dolfin.Constant(V0)

        self.measure = dV
        self.dS = dS

        self.tv_F = dmech.TimeVaryingConstant(
            val=f_val, val_ini=f_ini, val_fin=f_fin)
        F = self.tv_F.val

        self.res_form = - dolfin.inner(F, U_test)  * self.measure


        x = X + U

        x_tilde = x-x0

        rho0 = dolfin.assemble(Phis0*self.measure)


        self.res_form += dolfin.inner(( x * Phis0/dolfin.Constant(rho0) - x0/self.V0), x0_test) * self.measure



        self.tv_P0 = dmech.TimeVaryingConstant(
            val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
        P0 = self.tv_P0.val

        

        nf = dolfin.dot(N,dolfin.inv(kinematics.F))

        nf_norm = (1/2*dolfin.inner(nf,nf))**(1/2)

        n = nf/nf_norm

        # P = dolfin.dot(lbda, n)

        P = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x_tilde, n))

        self.res_form -= dolfin.inner(-P * n, U_test)  * nf_norm * kinematics.J * self.dS
        self.res_form += dolfin.inner(-P * n, lbda_test) * nf_norm * kinematics.J * self.dS
        self.res_form +=  dolfin.inner(F, lbda_test)  * self.measure
        self.res_form += dolfin.inner(-P *dolfin.cross(x_tilde, n), mu_test) * kinematics.J * nf_norm * self.dS
         

    def set_value_at_t_step(self,
            t_step):       
        self.tv_F.set_dvalue_at_t_step(t_step)
        self.tv_P0.set_value_at_t_step(t_step)
        pass


################################################################################

class SurfacePressureGradient0LoadingOperator(Operator):

    def __init__(self,
            x,
            x0,
            n,
            u_test,
            lbda,
            lbda_test,
            mu,
            mu_test,
            dS,
            dV,
            P0_val=None, P0_ini=None, P0_fin=None,
            f_val=None, f_ini=None, f_fin=None):


        self.dS = dS
        self.measure = dV
        
        # print([[dolfin.assemble((n[i]*n[j])*self.dS) for j in range(self.dim)] for i in range(self.dim)])
        # print("f_fin", f_fin)


        self.tv_f = dmech.TimeVaryingConstant(
            val=f_val, val_ini=f_ini, val_fin=f_fin)
        f = self.tv_f.val

        # gravity = -dolfin.assemble(f_fin[2]*self.measure)
        # print("gravity=", -dolfin.assemble(f_fin[2]*self.measure))

        self.res_form = - dolfin.inner(f, u_test) * self.measure

        self.tv_P0 = dmech.TimeVaryingConstant(
            val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
        P0 = self.tv_P0.val

        # print("dolfin.Constant([0.,0.,rhog[2]*dolfin.assemble(1*self.measure)]", [0.,0.,rhog[2]*dolfin.assemble(1*self.measure)])
        # lbda_ = dolfin.Constant([0.,0.,rhog[2]*dolfin.assemble(1*self.measure)])

        # self.tv_rhog = dmech.TimeVaryingConstant(
        #     val=None, val_ini=[0.,0.,0.], val_fin=rhog)
        # rhog = self.tv_rhog.val
        # dp  = -gravity/2*1e-4 #/dolfin.assemble(1*self.measure)#*dolfin.assemble(1*self.measure)/dolfin.assemble(1*self.dS)
        # lbda = dolfin.Constant([0,0,dp])
        # print("dp=", dp)


        x_tilde = x-dolfin.Constant(x0)
        
        
        # P = dolfin.dot(lbda,n)
        P = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x_tilde, n))
        # n = dolfin.Constant([0.,0.,1.])




        # print("P[0]", -dolfin.assemble(-dolfin.dot(lbda,n)*n[0]*self.dS))
        # print("P[1]", -dolfin.assemble(-dolfin.dot(lbda,n)*n[1]*self.dS))
        # print("P[2]", -dolfin.assemble(-dolfin.dot(lbda,n)*n[2]*self.dS))
           
        # self.res_form = ( - dolfin.inner(-P * n, u_test) + dolfin.inner(T_mu, mu_test) + dolfin.inner(rhog, lbda_test) + dolfin.inner(T_lbda, lbda_test)) * self.measure
        self.res_form -= dolfin.inner(-P * n, u_test) * self.dS
        self.res_form += dolfin.inner(dolfin.cross(x_tilde, -P * n), mu_test) * self.dS
        self.res_form += dolfin.inner(f, lbda_test) * self.measure
        self.res_form += dolfin.inner(-P * n, lbda_test) * self.dS



    def set_value_at_t_step(self,
            t_step):
        self.tv_f.set_value_at_t_step(t_step)
        self.tv_P0.set_value_at_t_step(t_step)
        


# ################################################################################

# class SurfacePressureGradient0LoadingOperator(Operator):

#     def __init__(self,
#             X,
#             X0,
#             u_test,
#             N,
#             vect,
#             measure,
#             X0_val=None, X0_ini=None, X0_fin=None,
#             P0_val=None, P0_ini=None, P0_fin=None):


#         self.measure=measure
        
#         self.tv_P0 = dmech.TimeVaryingConstant(
#             val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
#         P0 = self.tv_P0.val

#         self.tv_X0 = dmech.TimeVaryingConstant(
#             val=X0_val, val_ini=X0_ini, val_fin=X0_fin)
#         X0 = self.tv_X0.val

#         self.tv_lbda = dmech.TimeVaryingConstant(
#             val=None, val_ini=[0,0,0], val_fin=vect[0])
#         lbda = self.tv_lbda.val

#         self.tv_mu = dmech.TimeVaryingConstant(
#             val=None, val_ini=[0,0,0], val_fin=vect[1])
#         mu = self.tv_mu.val
        

#         P = P0 + dolfin.dot(lbda, N) + dolfin.dot(mu, dolfin.cross(X-X0, N))
           
#         self.res_form = - dolfin.inner(-P * N, u_test) * self.measure



#     def set_value_at_t_step(self,
#             t_step):

#         self.tv_X0.set_value_at_t_step(t_step)

#         self.tv_P0.set_value_at_t_step(t_step)
#         self.tv_lbda.set_value_at_t_step(t_step)
#         self.tv_mu.set_value_at_t_step(t_step)


# class CoefficientLambdaOperator1(Operator):

#     def __init__(self,
#         X,
#         x0,
#         U,
#         N,
#         kinematics,
#         lbda,
#         lbda_test,
#         mu,
#         P0,
#         measure):

#         self.measure = measure
        

#         x = X + U

#         nf = dolfin.dot(N,dolfin.inv(kinematics.F))

#         nf_norm = (1/2*dolfin.inner(nf,nf))**(1/2)

#         n = nf/nf_norm
        
#         self.tv_P0 = dmech.TimeVaryingConstant(
#             val=None, val_ini=0, val_fin=P0)
#         P0 = self.tv_P0.val


#         # P0 =dolfin.Constant(P0)        
#         P = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x-x0, n))

#         T = -P * n

#         self.res_form = - dolfin.inner(T, lbda_test) * nf_norm * kinematics.J * self.measure
         
    
#     def set_value_at_t_step(self,
#             t_step):       

#         self.tv_P0.set_value_at_t_step(t_step)
#         pass


# class CoefficientLambdaOperator2(Operator):

#     def __init__(self,
#         lbda,
#         lbda_test,
#         rhog,
#         measure):

        
#         self.measure = measure


#         self.tv_vforce = dmech.TimeVaryingConstant(
#             val=None, val_ini=[0.,0.,0.], val_fin=rhog)
#         vforce = self.tv_vforce.val

#         # T =  dolfin.Constant(rhog)
#         T = vforce

#         self.res_form = - dolfin.inner(T, lbda_test)  * self.measure
         

    
#     def set_value_at_t_step(self,
#             t_step):   

#         self.tv_vforce.set_value_at_t_step(t_step)
#         pass
        

# ################################################################################

# class CoefficientMuOperator(Operator):

#     def __init__(self,
#             X,
#             x0,
#             U,
#             N,
#             kinematics,
#             lbda,
#             mu,
#             mu_test,
#             P0,
#             measure):


#         self.measure = measure

#         x = X + U     

#         x_tilde = x-x0

#         nf = dolfin.dot(N,dolfin.inv(kinematics.F))
        
#         nf_norm = (1/2*dolfin.inner(nf,nf))**1/2

#         n = nf/nf_norm

#         self.tv_P0 = dmech.TimeVaryingConstant(
#             val=None, val_ini=0, val_fin=P0)
#         P0 = self.tv_P0.val

#         # P0 = dolfin.Constant(P0)

#         P = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x-x0, n))


#         T = -P *dolfin.cross(x_tilde, n)
#         self.res_form = - dolfin.inner(T, mu_test) * kinematics.J * nf_norm * self.measure


#     def set_value_at_t_step(self,
#             t_step):    
#         self.tv_P0.set_value_at_t_step(t_step)
#         pass



# ################################################################################


# class SurfacePressureGradientLoadingOperator(Operator):

#     def __init__(self,
#             X,
#             x0,
#             U,
#             U_test,
#             kinematics,
#             N,
#             lbda,
#             mu,
#             measure,
#             X0_val=None, X0_ini=None, X0_fin=None,
#             P0_val=None, P0_ini=None, P0_fin=None):
    

#         x = X + U

#         nf = dolfin.dot(N,dolfin.inv(kinematics.F))

#         nf_norm = (1/2*dolfin.inner(nf,nf))**1/2

#         n=nf/nf_norm

#         self.measure = measure


#         self.tv_P0 = dmech.TimeVaryingConstant(
#             val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
#         P0 = self.tv_P0.val

#         P = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x-x0, n))

#         T = -P * n
#         self.res_form = - dolfin.inner(T, U_test) * kinematics.J * nf_norm * self.measure
        




#     def set_value_at_t_step(self,
#             t_step):

#         # self.tv_x0.set_value_at_t_step(t_step)
        
#         self.tv_P0.set_value_at_t_step(t_step)
#         # self.tv_lbda.set_value_at_t_step(t_step)
#         # self.tv_mu.set_value_at_t_step(t_step)

################################################################################



# class CoefficientLambda0Operator1(Operator):

#     def __init__(self,
#         x,
#         x0,
#         n,
#         lbda,
#         lbda_test,
#         mu,
#         P0,
#         measure):


#         # self.tv_x0 =  dmech.TimeVaryingConstant(
#         #     val=None, val_ini=[0.,0.,0.], val_fin=x0)
#         # x0 = self.tv_x0.val
#         print("lbda1")
#         print("x0=", x0)
#         print("x=", x)
#         print("measure=", measure)

#         x_tilde = x-dolfin.Constant(x0)
#         self.measure = measure


#         # P0 = dolfin.Constant(P0)

#         print("P0", P0)

#         print("lbda, mu", lbda, mu)


#         self.tv_P0 = dmech.TimeVaryingConstant(
#             val=None, val_ini=0, val_fin=P0)
#         P0 = self.tv_P0.val

#         P = dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x_tilde, n))
#         # P = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x_tilde, n))

      
#         T = -P * n 
#         self.res_form = dolfin.inner(T, lbda_test)  * self.measure
         

    
    # def set_value_at_t_step(self,
    #         t_step):        
    #     # self.tv_P0.set_value_at_t_step(t_step)
    #     # self.tv_x0.set_value_at_t_step(t_step)
    #     pass


# class CoefficientLambda0Operator2(Operator):

#     def __init__(self,
#         x,
#         x0,
#         n,
#         lbda,
#         lbda_test,
#         rhog,
#         measure):

#         print("lbda2")
#         print("lbda, mu", lbda, lbda_test)

#         print("rhog=", rhog)
#         print("measure", measure)
#         self.measure = measure

    #     self.tv_rhog = dmech.TimeVaryingConstant(
    #         val=None, val_ini=[0.,0.,0.], val_fin=rhog)
    #     rhog = self.tv_rhog.val
        

    #     self.res_form = dolfin.inner(rhog, lbda_test)  * self.measure
         

    
    # def set_value_at_t_step(self,
    #         t_step):        
    #     self.tv_rhog.set_value_at_t_step(t_step)
    #     pass




################################################################################

# class CoefficientMu0Operator(Operator):

#     def __init__(self,
#             x,
#             x0,
#             n,
#             lbda,
#             mu,
#             mu_test,
#             P0,
#             measure):


#         print("mu")
#         print("x0=", x0)
#         print("measure", measure)
#         print("P0", P0)
#         print("lbda, mu", lbda, mu, mu_test)


        # x_tilde = x-dolfin.Constant(x0)
        # self.measure = measure


        # self.tv_P0 = dmech.TimeVaryingConstant(
        #     val=None, val_ini=0, val_fin=P0)
        # P0 = self.tv_P0.val

        # # P0 = dolfin.Constant(P0)

        # P = dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x_tilde, n))
        # # P = P0 + dolfin.dot(lbda, n) + dolfin.dot(mu, dolfin.cross(x_tilde, n))


        # T = -P *dolfin.cross(x_tilde, n)

        # self.res_form = dolfin.inner(T, mu_test) * self.measure


    # def set_value_at_t_step(self,
    #         t_step):        
    #     # self.tv_P0.set_value_at_t_step(t_step)
    #     pass
        
