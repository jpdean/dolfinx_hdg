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
#include "caster_petsc.h"

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
            [](Mat A,
               const std::vector<std::vector<std::shared_ptr<
                   const dolfinx::fem::Form<PetscScalar>>>> &a,
               const std::vector<std::shared_ptr<
                   const dolfinx::fem::DirichletBC<PetscScalar>>> &bcs)
            {
                  dolfinx_hdg::fem::assemble_matrix(
                      dolfinx::la::PETScMatrix::set_block_fn(A, ADD_VALUES), a, bcs);
            });

      m.def(
          "assemble_vector",
          [](pybind11::array_t<PetscScalar, pybind11::array::c_style> b,
             const dolfinx::fem::Form<PetscScalar> &L)
          {
                dolfinx_hdg::fem::assemble_vector<PetscScalar>(
                    xtl::span(b.mutable_data(), b.size()), L);
          },
          pybind11::arg("b"), pybind11::arg("L"),
          "Assemble linear form into an existing vector");
}
