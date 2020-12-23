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
import sys

import myPythonLibrary as mypy

import dolfin_mech as dmech

################################################################################

class TimeIntegrator():

    def __init__(self,
            problem,
            solver,
            parameters,
            print_out=True,
            print_sta=True,
            write_qois=True,
            write_sol=True,
            write_vtus=False,
            write_xmls=False):

        self.problem = problem

        self.solver = solver

        self.n_iter_for_accel = parameters["n_iter_for_accel"] if ("n_iter_for_accel" in parameters) else  4
        self.n_iter_for_decel = parameters["n_iter_for_decel"] if ("n_iter_for_decel" in parameters) else 16
        self.accel_coeff      = parameters["accel_coeff"]      if ("accel_coeff"      in parameters) else  2
        self.decel_coeff      = parameters["decel_coeff"]      if ("decel_coeff"      in parameters) else  2

        if (type(print_out) is str):
            if (print_out=="stdout"):
                self.printer_filename = None
            elif (print_out=="argv"):
                self.printer_filename = sys.argv[0][:-3]+".out"
            else:
                self.printer_filename = print_out+".out"
        else:
            self.printer_filename = None
        self.printer = mypy.Printer(
            filename=self.printer_filename,
            silent=not(print_out))
        self.solver.printer = self.printer

        if (type(print_sta) is str):
            if (print_sta=="stdout"):
                self.table_printer_filename = None
            elif (print_sta=="argv"):
                self.table_printer_filename = sys.argv[0][:-3]+".sta"
            else:
                self.table_printer_filename = print_sta+".sta"
        else:
            self.table_printer_filename = sys.argv[0][:-3]+".sta"
        self.table_printer = mypy.TablePrinter(
            titles=["k_step", "k_t", "dt", "t", "t_step", "n_iter", "success"],
            filename=self.table_printer_filename,
            silent=not(print_sta))

        self.write_qois = bool(write_qois) and (len(self.problem.qois)>0)
        if (self.write_qois):
            self.write_qois_filebasename = write_qois if (type(write_qois) is str) else sys.argv[0][:-3]+"-qois"

            self.qoi_printer = mypy.DataPrinter(
                names=["t"]+[qoi.name for qoi in self.problem.qois],
                filename=self.write_qois_filebasename+".dat")

            self.problem.update_qois()
            self.qoi_printer.write_line([0.]+[qoi.value for qoi in self.problem.qois])

        self.write_sol = bool(write_sol)
        if (self.write_sol):
            self.write_sol_filebasename = write_sol if (type(write_sol) is str) else sys.argv[0][:-3]+"-sol"

            self.functions_to_write  = []
            self.functions_to_write += self.problem.get_subsols_func_lst()
            self.functions_to_write += self.problem.get_subsols_func_old_lst()
            self.functions_to_write += self.problem.get_fois_func_lst()

            self.xdmf_file_sol = dmech.XDMFFile(
                filename=self.write_sol_filebasename+".xdmf",
                functions=self.functions_to_write)
            self.problem.update_fois()
            self.xdmf_file_sol.write(0.)

            self.write_vtus = bool(write_vtus)
            if (self.write_vtus):
                dmech.write_VTU_file(
                    filebasename=self.write_sol_filebasename,
                    function=self.problem.subsols["U"].subfunc,
                    time=0)

            self.write_xmls = bool(write_xmls)
            if (self.write_xmls):
                dolfin.File(self.write_sol_filebasename+"_"+str(0).zfill(3)+".xml") << self.problem.subsols["U"].subfunc



    def close(self):

        self.printer.close()
        self.table_printer.close()

        if (self.write_qois):
            self.qoi_printer.close()

        if (self.write_sol):
            self.xdmf_file_sol.close()



    def integrate(self):

        k_step = 0
        k_t_tot = 0
        n_iter_tot = 0
        self.printer.inc()
        for step in self.problem.steps:
            k_step += 1
            self.printer.print_var("k_step",k_step,-1)

            t = step.t_ini
            dt = step.dt_ini

            self.solver.constraints  = []
            self.solver.constraints += self.problem.constraints
            self.solver.constraints += step.constraints

            normal_penalties  = []
            normal_penalties += self.problem.normal_penalties
            normal_penalties += step.normal_penalties

            directional_penalties  = []
            directional_penalties += self.problem.directional_penalties
            directional_penalties += step.directional_penalties

            surface_tensions  = []
            surface_tensions += self.problem.surface_tensions
            surface_tensions += step.surface_tensions

            surface0_loadings  = []
            surface0_loadings += self.problem.surface0_loadings
            surface0_loadings += step.surface0_loadings

            pressure0_loadings  = []
            pressure0_loadings += self.problem.pressure0_loadings
            pressure0_loadings += step.pressure0_loadings

            volume0_loadings  = []
            volume0_loadings += self.problem.volume0_loadings
            volume0_loadings += step.volume0_loadings

            surface_loadings  = []
            surface_loadings += self.problem.surface_loadings
            surface_loadings += step.surface_loadings

            pressure_loadings  = []
            pressure_loadings += self.problem.pressure_loadings
            pressure_loadings += step.pressure_loadings

            volume_loadings  = []
            volume_loadings += self.problem.volume_loadings
            volume_loadings += step.volume_loadings

            # self.problem.set_variational_formulation(
            #     surface_tensions=surface_tensions,
            #     surface0_loadings=surface0_loadings,
            #     pressure0_loadings=pressure0_loadings,
            #     volume0_loadings=volume0_loadings,
            #     surface_loadings=surface_loadings,
            #     pressure_loadings=pressure_loadings,
            #     volume_loadings=volume_loadings)

            k_t = 0
            self.printer.inc()
            while (True):
                k_t += 1
                k_t_tot += 1
                self.printer.print_var("k_t",k_t,-1)

                if (t+dt > step.t_fin):
                    dt = step.t_fin - t
                self.printer.print_var("dt",dt)

                self.problem.set_variational_formulation(
                    normal_penalties=normal_penalties,
                    directional_penalties=directional_penalties,
                    surface_tensions=surface_tensions,
                    surface0_loadings=surface0_loadings,
                    pressure0_loadings=pressure0_loadings,
                    volume0_loadings=volume0_loadings,
                    surface_loadings=surface_loadings,
                    pressure_loadings=pressure_loadings,
                    volume_loadings=volume_loadings,
                    dt=dt)

                t += dt
                self.printer.print_var("t",t)

                t_step = (t - step.t_ini)/(step.t_fin - step.t_ini)
                self.printer.print_var("t_step",t_step)

                for constraint in step.constraints:
                    constraint.set_value_at_t_step(t_step)

                for loading in step.normal_penalties:
                    loading.set_value_at_t_step(t_step)

                for loading in step.directional_penalties:
                    loading.set_value_at_t_step(t_step)

                for loading in step.surface_tensions:
                    loading.set_value_at_t_step(t_step)

                for loading in step.surface0_loadings:
                    loading.set_value_at_t_step(t_step)

                for loading in step.pressure0_loadings:
                    loading.set_value_at_t_step(t_step)

                for loading in step.volume0_loadings:
                    loading.set_value_at_t_step(t_step)

                for loading in step.surface_loadings:
                    loading.set_value_at_t_step(t_step)

                for loading in step.pressure_loadings:
                    loading.set_value_at_t_step(t_step)

                for loading in step.volume_loadings:
                    loading.set_value_at_t_step(t_step)

                for inelastic_behavior in self.problem.inelastic_behaviors_internal:
                    inelastic_behavior.update_internal_variables_at_t(
                        t)

                self.problem.sol_old_func.vector()[:] = self.problem.sol_func.vector()[:]
                if (len(self.problem.subsols) > 1):
                    dolfin.assign(
                        self.problem.get_subsols_func_old_lst(),
                        self.problem.sol_old_func)
                solver_success, n_iter = self.solver.solve(k_step, k_t, dt, t)

                self.table_printer.write_line([k_step, k_t, dt, t, t_step, n_iter, solver_success])

                if (solver_success):
                    n_iter_tot += n_iter

                    if (self.write_sol):
                        self.problem.update_fois()
                        self.xdmf_file_sol.write(t)

                        if (self.write_vtus):
                            dmech.write_VTU_file(
                                filebasename=self.write_sol_filebasename,
                                function=self.problem.subsols["U"].subfunc,
                                time=k_t_tot)

                        if (self.write_xmls):
                            dolfin.File(self.write_sol_filebasename+"_"+str(k_t_tot).zfill(3)+".xml") << self.problem.subsols["U"].subfunc

                    if (self.write_qois):
                        self.problem.update_qois()
                        self.qoi_printer.write_line([t]+[qoi.value for qoi in self.problem.qois])

                    if dolfin.near(t, step.t_fin, eps=1e-9):
                        self.success = True
                        break
                    else:
                        if (n_iter <= self.n_iter_for_accel):
                            dt *= self.accel_coeff
                            if (dt > step.dt_max):
                                dt = step.dt_max
                        elif (n_iter >= self.n_iter_for_decel):
                            dt /= self.decel_coeff
                            if (dt < step.dt_min):
                                dt = step.dt_min
                else:
                    self.problem.sol_func.vector()[:] = self.problem.sol_old_func.vector()[:]
                    if (len(self.problem.subsols) > 1):
                        dolfin.assign(
                            self.problem.get_subsols_func_lst(),
                            self.problem.sol_func)

                    for inelastic_behavior in self.problem.inelastic_behaviors_internal:
                        inelastic_behavior.restore_old_value()

                    for constraint in step.constraints:
                        constraint.restore_old_value()

                    k_t -= 1
                    k_t_tot -= 1
                    t -= dt

                    dt /= self.decel_coeff
                    if (dt < step.dt_min):
                        self.printer.print_str("Warning! Time integrator failed to move forward!")
                        self.success = False
                        break

            self.printer.dec()

            if not (self.success):
                break

        self.printer.dec()

        return self.success
