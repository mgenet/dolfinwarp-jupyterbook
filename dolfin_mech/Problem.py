#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2020                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# from builtins import *

import collections
import dolfin
import numpy
import operator

import dolfin_mech as dmech

################################################################################

class Problem():



    def __init__(self):

        self.subsols = collections.OrderedDict()

        self.subdomains = []

        self.inelastic_behaviors_mixed    = []
        self.inelastic_behaviors_internal = []

        self.constraints           = []
        self.normal_penalties      = []
        self.directional_penalties = []

        self.surface_tensions   = []
        self.surface0_loadings  = []
        self.pressure0_loadings = []
        self.volume0_loadings   = []
        self.surface_loadings   = []
        self.pressure_loadings  = []
        self.volume_loadings    = []

        self.steps = []

        self.fois = []
        self.qois = []

        self.form_compiler_parameters = {}



    def set_mesh(self,
            mesh,
            compute_normals=False,
            compute_local_cylindrical_basis=False):

        self.dim = mesh.ufl_domain().geometric_dimension()

        self.mesh = mesh
        self.dV = dolfin.Measure(
            "dx",
            domain=self.mesh)
        self.mesh_V0 = dolfin.assemble(dolfin.Constant(1) * self.dV)

        if (compute_normals):
            self.mesh_normals = dolfin.FacetNormal(mesh)

        if (compute_local_cylindrical_basis):
            self.local_basis_fe = dolfin.VectorElement(
                family="DG",
                cell=mesh.ufl_cell(),
                degree=1)

            self.eR_expr = dolfin.Expression(
                ("+x[0]/sqrt(pow(x[0],2)+pow(x[1],2))", "+x[1]/sqrt(pow(x[0],2)+pow(x[1],2))"),
                element=self.local_basis_fe)
            self.eT_expr = dolfin.Expression(
                ("-x[1]/sqrt(pow(x[0],2)+pow(x[1],2))", "+x[0]/sqrt(pow(x[0],2)+pow(x[1],2))"),
                element=self.local_basis_fe)

            self.Q_expr = dolfin.as_matrix([[self.eR_expr[0], self.eR_expr[1]],
                                            [self.eT_expr[0], self.eT_expr[1]]])

            self.local_basis_fs = dolfin.FunctionSpace(
                mesh,
                self.local_basis_fe) # MG: element keyword don't work here…

            self.eR_func = dolfin.interpolate(
                v=self.eR_expr,
                V=self.local_basis_fs)
            self.eR_func.rename("eR", "eR")

            self.eT_func = dolfin.interpolate(
                v=self.eT_expr,
                V=self.local_basis_fs)
            self.eT_func.rename("eT", "eT")
        else:
            self.Q_expr = None



    def set_measures(self,
            domains=None,
            boundaries=None,
            points=None):

        self.dV = dolfin.Measure(
            "dx",
            domain=self.mesh,
            subdomain_data=domains)
        # if (domains is not None):
        #     self.dV = dolfin.Measure(
        #         "dx",
        #         domain=self.mesh,
        #         subdomain_data=domains)
        # else:
        #     self.dV = dolfin.Measure(
        #         "dx",
        #         domain=self.mesh)

        self.dS = dolfin.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=boundaries)
        # if (boundaries is not None):
        #     self.dS = dolfin.Measure(
        #         "ds",
        #         domain=self.mesh,
        #         subdomain_data=boundaries)
        # else:
        #     self.dS = dolfin.Measure(
        #         "ds",
        #         domain=self.mesh)

        self.dP = dolfin.Measure(
            "dP",
            domain=self.mesh,
            subdomain_data=points)
        # if (points is not None):
        #     self.dP = dolfin.Measure(
        #         "dP",
        #         domain=self.mesh,
        #         subdomain_data=points)
        # else:
        #     self.dP = dolfin.Measure(
        #         "dP",
        #         domain=self.mesh)



    def add_subsol(self,
            name,
            *args,
            **kwargs):

        subsol = dmech.SubSol(
            name=name,
            *args,
            **kwargs)
        self.subsols[name] = subsol
        return subsol



    def add_scalar_subsol(self,
            name,
            family="CG",
            degree=1,
            init_val=None):

        fe = dolfin.FiniteElement(
            family=family,
            cell=self.mesh.ufl_cell(),
            degree=degree)

        self.add_subsol(
            name=name,
            fe=fe,
            init_val=init_val)



    def add_vector_subsol(self,
            name,
            family="CG",
            degree=1,
            init_val=None):

        fe = dolfin.VectorElement(
            family=family,
            cell=self.mesh.ufl_cell(),
            degree=degree)

        self.add_subsol(
            name=name,
            fe=fe,
            init_val=init_val)



    def add_tensor_subsol(self,
            name,
            family="CG",
            degree=1,
            init_val=None):

        fe = dolfin.TensorElement(
            family=family,
            cell=self.mesh.ufl_cell(),
            degree=degree)

        self.add_subsol(
            name=name,
            fe=fe,
            init_val=init_val)



    def set_solution_finite_element(self):

        if (len(self.subsols) == 1):
            self.sol_fe = list(self.subsols.values())[0].fe
        else:
            self.sol_fe = dolfin.MixedElement([subsol.fe for subsol in self.subsols.values()])
        #print(self.sol_fe)



    def set_solution_function_space(self):

        self.sol_fs = dolfin.FunctionSpace(
            self.mesh,
            self.sol_fe) # MG: element keyword don't work here…
        #print(self.sol_fs)



    def get_subsol_function_space(self,
            name):

        index = list(self.subsols.keys()).index(name)
        # print(str(name)+" index = "+str(index))
        return self.sol_fs.sub(index)



    def set_solution_functions(self):

        self.sol_func     = dolfin.Function(self.sol_fs)
        self.sol_old_func = dolfin.Function(self.sol_fs)
        self.dsol_func    = dolfin.Function(self.sol_fs)
        self.dsol_test    = dolfin.TestFunction(self.sol_fs)
        self.dsol_tria    = dolfin.TrialFunction(self.sol_fs)

        if (len(self.subsols) == 1):
            subfuncs  = (self.sol_func,)
            dsubtests = (self.dsol_test,)
            dsubtrias = (self.dsol_tria,)
            funcs     = (self.sol_func,)
            funcs_old = (self.sol_old_func,)
            dfuncs    = (self.dsol_func,)
        else:
            subfuncs  = dolfin.split(self.sol_func)
            dsubtests = dolfin.split(self.dsol_test)
            dsubtrias = dolfin.split(self.dsol_tria)
            funcs     = dolfin.Function(self.sol_fs).split(deepcopy=1)
            funcs_old = dolfin.Function(self.sol_fs).split(deepcopy=1)
            dfuncs    = dolfin.Function(self.sol_fs).split(deepcopy=1)

        for (k_subsol,subsol) in enumerate(self.subsols.values()):
            subsol.subfunc  = subfuncs[k_subsol]
            subsol.dsubtest = dsubtests[k_subsol]
            subsol.dsubtria = dsubtrias[k_subsol]

            subsol.func = funcs[k_subsol]
            subsol.func.rename(subsol.name, subsol.name)
            subsol.func_old = funcs_old[k_subsol]
            subsol.func_old.rename(subsol.name+"_old", subsol.name+"_old")
            subsol.dfunc = dfuncs[k_subsol]
            subsol.dfunc.rename("d"+subsol.name, "d"+subsol.name)

        init_val = [str(val) for val in numpy.concatenate([subsol.init_val.flatten() for subsol in self.subsols.values()])]
        # print("init_val = "+str(init_val))
        if (init_val == ["0.0"]):
            init_val = "0.0"
            # print("init_val = "+str(init_val))
        self.sol_func.interpolate(dolfin.Expression(
            init_val,
            element=self.sol_fe))
        self.sol_old_func.interpolate(dolfin.Expression(
            init_val,
            element=self.sol_fe))
        if (len(self.subsols) > 1):
            dolfin.assign(
                self.get_subsols_func_lst(),
                self.sol_func)
            dolfin.assign(
                self.get_subsols_func_old_lst(),
                self.sol_old_func)



    def get_subsols_func_lst(self):

        return [subsol.func for subsol in self.subsols.values()]



    def get_subsols_func_old_lst(self):

        return [subsol.func_old for subsol in self.subsols.values()]



    def get_subsols_dfunc_lst(self):

        return [subsol.dfunc for subsol in self.subsols.values()]



    def set_quadrature_degree(self,
            quadrature_degree):

        self.form_compiler_parameters["quadrature_degree"] = quadrature_degree



    def set_foi_finite_elements_DG(self,
            degree=0): # MG20180420: DG elements are simpler to manage than quadrature elements, since quadrature elements must be compatible with the expression's degree, which is not always trivial (e.g., for J…)

        self.sfoi_fe = dolfin.FiniteElement(
            family="DG",
            cell=self.mesh.ufl_cell(),
            degree=degree)

        self.vfoi_fe = dolfin.VectorElement(
            family="DG",
            cell=self.mesh.ufl_cell(),
            degree=degree)

        self.mfoi_fe = dolfin.TensorElement(
            family="DG",
            cell=self.mesh.ufl_cell(),
            degree=degree)



    def set_foi_finite_elements_Quad(self,
            degree=0): # MG20180420: DG elements are simpler to manage than quadrature elements, since quadrature elements must be compatible with the expression's degree, which is not always trivial (e.g., for J…)

        self.sfoi_fe = dolfin.FiniteElement(
            family="Quadrature",
            cell=self.mesh.ufl_cell(),
            degree=degree,
            quad_scheme="default")
        self.sfoi_fe._quad_scheme = "default"           # MG20180406: is that even needed?
        for sub_element in self.sfoi_fe.sub_elements(): # MG20180406: is that even needed?
            sub_element._quad_scheme = "default"        # MG20180406: is that even needed?

        self.vfoi_fe = dolfin.VectorElement(
            family="Quadrature",
            cell=self.mesh.ufl_cell(),
            degree=degree,
            quad_scheme="default")
        self.vfoi_fe._quad_scheme = "default"           # MG20180406: is that even needed?
        for sub_element in self.vfoi_fe.sub_elements(): # MG20180406: is that even needed?
            sub_element._quad_scheme = "default"        # MG20180406: is that even needed?

        self.mfoi_fe = dolfin.TensorElement(
            family="Quadrature",
            cell=self.mesh.ufl_cell(),
            degree=degree,
            quad_scheme="default")
        self.mfoi_fe._quad_scheme = "default"           # MG20180406: is that still needed?
        for sub_element in self.mfoi_fe.sub_elements(): # MG20180406: is that still needed?
            sub_element._quad_scheme = "default"        # MG20180406: is that still needed?



    def set_foi_function_spaces(self):

        self.sfoi_fs = dolfin.FunctionSpace(
            self.mesh,
            self.sfoi_fe) # MG: element keyword don't work here…

        self.vfoi_fs = dolfin.FunctionSpace(
            self.mesh,
            self.vfoi_fe) # MG: element keyword don't work here…

        self.mfoi_fs = dolfin.FunctionSpace(
            self.mesh,
            self.mfoi_fe) # MG: element keyword don't work here…



    def add_foi(self,
            *args,
            **kwargs):

        foi = dmech.FOI(
            *args,
            form_compiler_parameters=self.form_compiler_parameters,
            **kwargs)
        self.fois += [foi]
        return foi



    def update_fois(self):

        for foi in self.fois:
            foi.update()



    def get_fois_func_lst(self):

        return [foi.func for foi in self.fois]



    def add_qoi(self,
            *args,
            **kwargs):

        qoi = dmech.QOI(
            *args,
            form_compiler_parameters=self.form_compiler_parameters,
            **kwargs)
        self.qois += [qoi]
        return qoi



    def update_qois(self):

        for qoi in self.qois:
            qoi.update()



    def add_constraint(self,
            *args,
            **kwargs):

        constraint = dmech.Constraint(
            *args,
            **kwargs)
        self.constraints += [constraint]
        return constraint



    def add_normal_penalty(self,
            *args,
            **kwargs):

        loading = dmech.Loading(
            *args,
            **kwargs)
        self.normal_penalties += [loading]
        return loading



    def add_directional_penalty(self,
            *args,
            **kwargs):

        loading = dmech.Loading(
            *args,
            **kwargs)
        self.directional_penalties += [loading]
        return loading



    def add_surface_tension(self,
            *args,
            **kwargs):

        loading = dmech.Loading(
            *args,
            **kwargs)
        self.surface_tensions += [loading]
        return loading



    def add_surface0_loading(self,
            *args,
            **kwargs):

        loading = dmech.Loading(
            *args,
            **kwargs)
        self.surface0_loadings += [loading]
        return loading



    def add_pressure0_loading(self,
            *args,
            **kwargs):

        loading = dmech.Loading(
            *args,
            **kwargs)
        self.pressure0_loadings += [loading]
        return loading



    def add_volume0_loading(self,
            *args,
            **kwargs):

        loading = dmech.Loading(
            *args,
            **kwargs)
        self.volume0_loadings += [loading]
        return loading



    def add_surface_loading(self,
            *args,
            **kwargs):

        loading = dmech.Loading(
            *args,
            **kwargs)
        self.surface_loadings += [loading]
        return loading



    def add_pressure_loading(self,
            *args,
            **kwargs):

        loading = dmech.Loading(
            *args,
            **kwargs)
        self.pressure_loadings += [loading]
        return loading



    def add_volume_loading(self,
            *args,
            **kwargs):

        loading = dmech.Loading(
            *args,
            **kwargs)
        self.volume_loadings += [loading]
        return loading



    def add_step(self,
            Deltat=1.,
            *args,
            **kwargs):

        if len(self.steps) == 0:
            t_ini = 0.
            t_fin = Deltat
        else:
            t_ini = self.steps[-1].t_fin
            t_fin = t_ini + Deltat
        step = dmech.Step(
            t_ini=t_ini,
            t_fin=t_fin,
            *args,
            **kwargs)
        self.steps += [step]
        return step
