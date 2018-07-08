#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018                                            ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_cm as dcm

################################################################################

class Step():



    def __init__(self,
            t_ini=0.,
            t_fin=1.,
            dt_ini=None,
            dt_min=None,
            dt_max=None,
            constraints=None, #MG20180508: Do not use list as default value because it is static
            penalties=None,
            surface0_loadings=None,
            pressure0_loadings=None,
            volume0_loadings=None,
            surface_loadings=None,
            pressure_loadings=None,
            volume_loadings=None):

        self.t_ini = t_ini
        self.t_fin = t_fin

        self.dt_ini = dt_ini if (dt_ini is not None) else t_fin - t_ini
        self.dt_min = dt_min if (dt_min is not None) else self.dt_ini
        self.dt_max = dt_max if (dt_max is not None) else self.dt_ini

        self.constraints        = constraints        if (constraints        is not None) else []
        self.penalties          = penalties          if (penalties          is not None) else []
        self.surface0_loadings  = surface0_loadings  if (surface0_loadings  is not None) else []
        self.pressure0_loadings = pressure0_loadings if (pressure0_loadings is not None) else []
        self.volume0_loadings   = volume0_loadings   if (volume0_loadings   is not None) else []
        self.surface_loadings   = surface_loadings   if (surface_loadings   is not None) else []
        self.pressure_loadings  = pressure_loadings  if (pressure_loadings  is not None) else []
        self.volume_loadings    = volume_loadings    if (volume_loadings    is not None) else []



    def add_constraint(self,
            *args,
            **kwargs):

        constraint = dcm.Constraint(
            *args,
            **kwargs)
        self.constraints += [constraint]
        return constraint



    def add_penalty(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.penalties += [loading]
        return loading



    def add_surface0_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.surface0_loadings += [loading]
        return loading



    def add_pressure0_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.pressure0_loadings += [loading]
        return loading



    def add_volume0_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.volume0_loadings += [loading]
        return loading



    def add_surface_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.surface_loadings += [loading]
        return loading



    def add_pressure_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.pressure_loadings += [loading]
        return loading



    def add_volume_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.volume_loadings += [loading]
        return loading
