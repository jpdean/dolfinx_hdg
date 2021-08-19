#pragma once

// FIXME See which of these aren't needed
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <iterator>
#include <vector>
#include <xtl/xspan.hpp>
#include <dolfinx/common/array2d.h>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> // To cout xtensor arrays
#include <xtensor-blas/xlinalg.hpp>

namespace dolfinx_hdg::sc
{
    std::vector<double> get_cell_coord_dofs(
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
        const int cell);

    std::vector<double> get_facet_coord_dofs(
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
        const int f);

    // template <typename T>
    // xt::xarray<double> assemble_cell_vector(
    //     const dolfinx::fem::Form<T> &L,
    //     const int cell,
    //     const tcb::span<const int> &cell_facets,
    //     const xtl::span<const T> &constants,
    //     const dolfinx::array2d<T> &coeffs)
    // {
    // }

    // NOTE This approach reuses code and makes for a simpler
    // implementation, but means looping over facets more that
    // once. All cell matrices could be computed in one go
    template <typename T>
    xt::xarray<double> assemble_cell_matrix(
        const dolfinx::fem::Form<T> &a,
        const int cell,
        const tcb::span<const int> &cell_facets,
        const xtl::span<const T> &constants,
        const dolfinx::array2d<T> &coeffs)
    {
        // FIXME Vector elements (i.e. block size \neq 1) might break some of this
        // TODO Pass just dofmap?
        // TODO Dof transformation stuff
        // FIXME Think of how to do this nicely. Should use integral_ids
        // FIXME TODO Permutations. See dolfinx exterior facet integrals
        // TODO Active cells. NOTE For the facet space, getting the active
        // cells in the usual way below actually gets the active facet
        // numbers, as the facets are treated as cells
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a.mesh();
        assert(mesh);
        const int tdim = mesh->topology().dim();
        std::vector<double> coordinate_dofs =
            get_cell_coord_dofs(mesh, cell);
        // NOTE These are created outside, but check mesh isn't different
        // for each form!
        const std::vector<std::uint8_t> &perms =
            mesh->topology().get_facet_permutations();

        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap_0 =
            a.function_spaces().at(0)->dofmap()->list();
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap_1 =
            a.function_spaces().at(1)->dofmap()->list();
        const int bs_0 = a.function_spaces().at(0)->dofmap()->bs();
        const int bs_1 = a.function_spaces().at(1)->dofmap()->bs();
        const int num_dofs_0 = bs_0 * dofmap_0.links(0).size();
        const int num_dofs_1 = bs_1 * dofmap_1.links(0).size();
        const int codim_0 = a.function_spaces().at(0)->codimension();
        const int codim_1 = a.function_spaces().at(1)->codimension();

        // TODO Check codim == 0 or 1
        const int num_rows =
            codim_0 == 0 ? num_dofs_0 : cell_facets.size() * num_dofs_0;
        const int num_cols =
            codim_1 == 0 ? num_dofs_1 : cell_facets.size() * num_dofs_1;
        xt::xarray<double> Ae = xt::zeros<double>({num_rows, num_cols});

        // If we have a cell-cell form, call cell kernel here
        // FIXME Integral
        if (codim_0 == 0 && codim_1 == 0)
        {
            for (int i : a.integral_ids(dolfinx::fem::IntegralType::cell))
            {
                if (i == -1)
                {
                    const auto &kernel =
                        a.kernel(dolfinx::fem::IntegralType::cell, i);
                    kernel(Ae.data(), coeffs.row(cell).data(), constants.data(),
                           coordinate_dofs.data(), nullptr, nullptr);
                }
            }
        }

        for (int local_f = 0; local_f < cell_facets.size(); ++local_f)
        {
            const int f = cell_facets[local_f];
            // If we have a cell-cell form, call facet kernel here
            if (codim_0 == 0 && codim_1 == 0)
            {
                for (int i : a.integral_ids(
                         dolfinx::fem::IntegralType::exterior_facet))
                {
                    if (i == -1)
                    {
                        const auto &kernel =
                            a.kernel(dolfinx::fem::IntegralType::exterior_facet, i);
                        kernel(Ae.data(), coeffs.row(cell).data(), constants.data(),
                               coordinate_dofs.data(), &local_f,
                               &perms[cell * cell_facets.size() + local_f]);
                    }
                }
            }
            else
            {
                // If not a cell-cell exterior facet integral, we need to compute
                // the facet matrix and put in correct place in "cell" matrix.
                xt::xarray<double> Ae_f = xt::zeros<double>({num_dofs_0,
                                                             num_dofs_1});
                for (int i : a.integral_ids(
                         dolfinx::fem::IntegralType::exterior_facet))
                {
                    // One codim must be zero to carry out an exterior_facet
                    // integral
                    assert(codim_0 == 0 || codim_1 == 0);
                    if (i == -1)
                    {
                        const auto &kernel =
                            a.kernel(dolfinx::fem::IntegralType::exterior_facet, i);
                        kernel(Ae_f.data(), coeffs.row(cell).data(), constants.data(),
                               coordinate_dofs.data(), &local_f,
                               &perms[cell * cell_facets.size() + local_f]);
                    }
                }
                for (int i : a.integral_ids(
                         dolfinx::fem::IntegralType::cell))
                {
                    // Both codims must be 1 to carry out an cell integral here
                    assert(codim_0 == 1 && codim_1 == 1);
                    if (i == -1)
                    {
                        std::vector<double> fact_coordinate_dofs =
                            get_facet_coord_dofs(mesh, f);
                        const auto &kernel =
                            a.kernel(dolfinx::fem::IntegralType::cell, i);
                        kernel(Ae_f.data(), coeffs.row(cell).data(), constants.data(),
                               fact_coordinate_dofs.data(), nullptr, nullptr);
                    }
                }
                const int start_row =
                    codim_0 == 0 ? 0 : local_f * num_dofs_0;
                const int end_row =
                    codim_0 == 0 ? num_rows : start_row + num_dofs_0;

                const int start_col =
                    codim_1 == 0 ? 0 : local_f * num_dofs_1;
                const int end_col =
                    codim_1 == 0 ? num_cols : start_col + num_dofs_1;

                xt::view(Ae,
                         xt::range(start_row, end_row),
                         xt::range(start_col, end_col)) = Ae_f;
            }
        }

        return Ae;
    }
}
