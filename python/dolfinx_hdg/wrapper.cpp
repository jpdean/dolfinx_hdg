#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <dolfinx_hdg/hello.h>
#include <dolfinx_hdg/utils.h>
#include <dolfinx_hdg/petsc.h>
#include <petscmat.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx_hdg/assembler.h>
#include <dolfinx/la/PETScMatrix.h>
#include <iostream>
#include "caster_petsc.h"

// FIXME Include this from dolfinx wrappers
namespace dolfinx_hdg_wrappers
{
    template <typename T>
    std::map<std::pair<dolfinx::fem::IntegralType, int>,
            std::pair<xtl::span<const T>, int>>
    py_to_cpp_coeffs(const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                                    py::array_t<T, py::array::c_style>>& coeffs)
    {
        using Key_t = typename std::remove_reference_t<decltype(coeffs)>::key_type;
        std::map<Key_t, std::pair<xtl::span<const T>, int>> c;
        std::transform(coeffs.cbegin(), coeffs.cend(), std::inserter(c, c.end()),
                        [](auto& e) -> typename decltype(c)::value_type
                        {
                        return {
                            e.first,
                            {xtl::span<const T>(e.second.data(), e.second.size()),
                                e.second.shape(1)}};
                        });
        return c;
    }
}

PYBIND11_MODULE(cpp, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("say_hello", &hello::say_hello, "A function which says hello");

    m.def("create_matrix", dolfinx_hdg::fem::create_matrix,
          pybind11::return_value_policy::take_ownership, pybind11::arg("a"),
          pybind11::arg("type") = std::string(),
          "Create a PETSc Mat for bilinear form.");

    // m.def("create_sparsity_pattern",
    //       &dolfinx_hdg::fem::create_sparsity_pattern<PetscScalar>,
    //       "Create a sparsity pattern for bilinear form.");

    m.def("assemble_matrix_petsc",
          [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
             const py::array_t<PetscScalar, py::array::c_style>& constants,
             const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                            py::array_t<PetscScalar, py::array::c_style>>&
                coefficients,
             const std::vector<std::shared_ptr<
                              const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs)
          {
            auto _coefficients = dolfinx_hdg_wrappers::py_to_cpp_coeffs(coefficients);
            dolfinx_hdg::fem::assemble_matrix(
                  dolfinx::la::PETScMatrix::set_block_fn(A, ADD_VALUES), a,
                  xtl::span(constants), _coefficients, bcs);
          });

    // TODO / FIXME dolfinx uses templating for generic type rather than PetscScalar
    // here (and elsewhere). Add this.
    m.def(
        "assemble_vector",
        [](pybind11::array_t<PetscScalar, pybind11::array::c_style> b,
           const dolfinx::fem::Form<PetscScalar> &L,
           const py::array_t<PetscScalar, py::array::c_style>& constants,
           const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<PetscScalar, py::array::c_style>>& coefficients)
        {
            auto _coefficients = dolfinx_hdg_wrappers::py_to_cpp_coeffs(coefficients);
            dolfinx_hdg::fem::assemble_vector<PetscScalar>(
                xtl::span(b.mutable_data(), b.size()), L, constants,
                _coefficients);
        },
        pybind11::arg("b"), pybind11::arg("L"), py::arg("constants"),
        py::arg("coeffs"),
        "Assemble linear form into an existing vector");

    // m.def(
    //     "back_sub",
    //     [](pybind11::array_t<PetscScalar, pybind11::array::c_style> x,
    //        pybind11::array_t<PetscScalar, pybind11::array::c_style> xbar,
    //        const std::vector<std::vector<std::shared_ptr<
    //            const dolfinx::fem::Form<PetscScalar>>>> &a,
    //        const std::vector<std::shared_ptr<
    //            const dolfinx::fem::Form<PetscScalar>>> &L)
    //     {
    //         // FIXME xbar is constant. How do I pass it as a const?
    //         dolfinx_hdg::sc::back_sub<PetscScalar>(
    //             xtl::span(x.mutable_data(), x.size()),
    //             xtl::span(xbar.mutable_data(), xbar.size()),
    //             a, L);
    //     },
    //     pybind11::arg("x"), pybind11::arg("xbar"), pybind11::arg("a"),
    //     pybind11::arg("L"),
    //     "Perform backsubstitution");
}
