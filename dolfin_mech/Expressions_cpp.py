#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Colin Laville, 2021-2022                                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

def get_ExprMeshFunction_cpp_pybind():

    cpp_code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

class MeshExpr : public dolfin::Expression
{
public:
    // Create scalar expression
    MeshExpr() : dolfin::Expression() {}

    // Function for evaluating expression on each cell
    void eval(
        Eigen::Ref<Eigen::VectorXd> values,
        Eigen::Ref<const Eigen::VectorXd> x,
        const ufc::cell& cell) const override
    {
        const uint cell_index = cell.index;
        values[0] = (*f)[cell_index];
    }

    // The data stored in mesh functions
    std::shared_ptr<dolfin::MeshFunction<double>> f;
};

PYBIND11_MODULE(SIGNATURE, m)
{
pybind11::class_<MeshExpr, std::shared_ptr<MeshExpr>, dolfin::Expression>
(m, "MeshExpr")
.def(pybind11::init<>())
.def_readwrite("f", &MeshExpr::f);
}
"""

    return cpp_code
