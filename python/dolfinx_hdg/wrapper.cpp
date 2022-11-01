#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <dolfinx_hdg/utils.h>
#include <dolfinx_hdg/petsc.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx_hdg/assembler.h>
#include <dolfinx/la/petsc.h>
#include "caster_petsc.h"

// FIXME Include this from dolfinx wrappers
namespace dolfinx_hdg_wrappers
{
    template <typename T>
    std::map<std::pair<dolfinx::fem::IntegralType, int>,
             std::pair<std::span<const T>, int>>
    py_to_cpp_coeffs(const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                                    py::array_t<T, py::array::c_style>> &coeffs)
    {
        using Key_t = typename std::remove_reference_t<decltype(coeffs)>::key_type;
        std::map<Key_t, std::pair<std::span<const T>, int>> c;
        std::transform(coeffs.begin(), coeffs.end(), std::inserter(c, c.end()),
                       [](auto &e) -> typename decltype(c)::value_type
                       {
                           return {e.first,
                                   {std::span(e.second.data(), e.second.size()),
                                    e.second.shape(1)}};
                       });
        return c;
    }

    /// Create an n-dimensional py::array_t that shares data with a
    /// std::vector. The std::vector owns the data, and the py::array_t
    /// object keeps the std::vector alive.
    /// From https://github.com/pybind/pybind11/issues/1042
    template <typename Sequence, typename U>
    py::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq, U &&shape)
    {
        auto data = seq.data();
        std::unique_ptr<Sequence> seq_ptr = std::make_unique<Sequence>(std::move(seq));
        auto capsule = py::capsule(
            seq_ptr.get(), [](void *p)
            { std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p)); });
        seq_ptr.release();
        return py::array(shape, data, capsule);
    }

    /// Create a py::array_t that shares data with a std::vector. The
    /// std::vector owns the data, and the py::array_t object keeps the std::vector
    /// alive.
    // From https://github.com/pybind/pybind11/issues/1042
    template <typename Sequence>
    py::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq)
    {
        return as_pyarray(std::move(seq), std::array{seq.size()});
    }

    template <typename T>
    void declare_functions(py::module &m)
    {
        m.def(
            "assemble_vector",
            [](py::array_t<T, py::array::c_style> b, const dolfinx::fem::Form<T> &L,
               const py::array_t<T, py::array::c_style> &constants,
               const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                              py::array_t<T, py::array::c_style>> &coefficients)
            {
                dolfinx_hdg::fem::assemble_vector<T>(
                    std::span(b.mutable_data(), b.size()), L,
                    std::span(constants.data(), constants.size()),
                    dolfinx_hdg_wrappers::py_to_cpp_coeffs(coefficients));
            },
            py::arg("b"), py::arg("L"), py::arg("constants"), py::arg("coeffs"),
            "Assemble linear form into an existing vector with pre-packed constants "
            "and coefficients");

        m.def(
            "pack_coefficients",
            [](const dolfinx::fem::Form<T> &form)
            {
                using Key_t = typename std::pair<dolfinx::fem::IntegralType, int>;

                // Pack coefficients
                std::map<Key_t, std::pair<std::vector<T>, int>> coeffs = dolfinx_hdg::fem::allocate_coefficient_storage(form);
                dolfinx_hdg::fem::pack_coefficients(form, coeffs);

                // Move into NumPy data structures
                std::map<Key_t, py::array_t<T, py::array::c_style>> c;
                std::transform(
                    coeffs.begin(), coeffs.end(), std::inserter(c, c.end()),
                    [](auto &e) -> typename decltype(c)::value_type
                    {
                        int num_ents = e.second.first.empty()
                                           ? 0
                                           : e.second.first.size() / e.second.second;
                        return {e.first, dolfinx_hdg_wrappers::as_pyarray(
                                             std::move(e.second.first),
                                             std::array{num_ents, e.second.second})};
                    });

                return c;
            },
            py::arg("form"), "Pack coefficients for a Form.");
    }
}

PYBIND11_MODULE(cpp, m)
{
    m.doc() = "Custom assemblers for HDG"; // optional module docstring

    m.def("create_matrix", dolfinx_hdg::fem::create_matrix,
          pybind11::return_value_policy::take_ownership, pybind11::arg("a"),
          pybind11::arg("type") = std::string(),
          "Create a PETSc Mat for bilinear form.");

    m.def(
        "assemble_matrix",
        [](Mat A, const dolfinx::fem::Form<PetscScalar> &a,
           const py::array_t<PetscScalar, py::array::c_style> &constants,
           const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                          py::array_t<PetscScalar, py::array::c_style>> &
               coefficients,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>> &bcs,
           bool unrolled)
        {
            if (unrolled)
            {
                auto set_fn = dolfinx::la::petsc::Matrix::set_block_expand_fn(
                    A, a.function_spaces()[0]->dofmap()->bs(),
                    a.function_spaces()[1]->dofmap()->bs(), ADD_VALUES);
                dolfinx_hdg::fem::assemble_matrix(
                    set_fn, a, std::span(constants.data(), constants.size()),
                    dolfinx_hdg_wrappers::py_to_cpp_coeffs(coefficients), bcs);
            }
            else
            {
                dolfinx_hdg::fem::assemble_matrix(
                    dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), a,
                    std::span(constants.data(), constants.size()),
                    dolfinx_hdg_wrappers::py_to_cpp_coeffs(coefficients), bcs);
            }
        },
        py::arg("A"), py::arg("a"), py::arg("constants"), py::arg("coeffs"),
        py::arg("bcs"), py::arg("unrolled") = false,
        "Assemble bilinear form into an existing PETSc matrix");

    m.def("create_matrix_block", &dolfinx_hdg::fem::create_matrix_block,
          py::return_value_policy::take_ownership, py::arg("a"),
          py::arg("type") = std::string(),
          "Create monolithic sparse matrix for stacked bilinear forms.");

    dolfinx_hdg_wrappers::declare_functions<double>(m);
    dolfinx_hdg_wrappers::declare_functions<float>(m);
    dolfinx_hdg_wrappers::declare_functions<std::complex<double>>(m);
    dolfinx_hdg_wrappers::declare_functions<std::complex<float>>(m);
}
