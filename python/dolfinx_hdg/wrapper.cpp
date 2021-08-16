#include <pybind11/pybind11.h>
#include <dolfinx_hdg/hello.h>
#include <dolfinx_hdg/utils.h>

PYBIND11_MODULE(cpp, m)
{
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("say_hello", &hello::say_hello, "A function which says hello");

  // m.def("create_matrix", dolfinx_hdg::fem::create_matrix,
  //       pybind11::return_value_policy::take_ownership, pybind11::arg("a"),
  //       pybind11::arg("type") = std::string(),
  //       "Create a PETSc Mat for bilinear form.");

  m.def("create_sparsity_pattern",
        &dolfinx_hdg::fem::create_sparsity_pattern<PetscScalar>,
        "Create a sparsity pattern for bilinear form.");
}
