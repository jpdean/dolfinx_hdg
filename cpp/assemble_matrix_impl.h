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

namespace dolfinx_hdg::fem::impl
{
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
        // TODO Pass just dofmap?
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a.mesh();
        assert(mesh);
        const int tdim = mesh->topology().dim();
        const dolfinx::graph::AdjacencyList<std::int32_t> &x_dofmap =
            mesh->geometry().dofmap();
        const std::size_t num_dofs_g = x_dofmap.num_links(cell);
        const xt::xtensor<double, 2> &x_g = mesh->geometry().x();
        std::vector<double> coordinate_dofs(3 * num_dofs_g);
        auto x_dofs = x_dofmap.links(cell);
        for (std::size_t i = 0; i < x_dofs.size(); ++i)
        {
            std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                        std::next(coordinate_dofs.begin(), 3 * i));
        }
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
                        // FIXME Get facet coord dofs properly and check
                        auto facet_x_dofs = mesh->topology().connectivity(tdim - 1, tdim - 2)->links(f);
                        std::vector<double> fact_coordinate_dofs(3 * mesh->topology().connectivity(tdim - 1, tdim - 2)->num_links(0));
                        for (std::size_t i = 0; i < facet_x_dofs.size(); ++i)
                        {
                            std::copy_n(xt::row(x_g, facet_x_dofs[i]).begin(), 3,
                                        std::next(fact_coordinate_dofs.begin(), 3 * i));
                        }
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

    template <typename T>
    void assemble_matrix(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_set,
        const std::vector<std::vector<std::shared_ptr<
            const dolfinx::fem::Form<T>>>> &a,
        const xtl::span<const T> &constants,
        const dolfinx::array2d<T> &coeffs, const std::vector<bool> &bc0,
        const std::vector<bool> &bc1)
    {
        // FIXME Vector elements (i.e. block size \neq 1) might break some of this
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a[0][0]->mesh();
        assert(mesh);
        const int tdim = mesh->topology().dim();

        const dolfinx::graph::AdjacencyList<std::int32_t> &x_dofmap =
            mesh->geometry().dofmap();
        const std::size_t num_dofs_g = x_dofmap.num_links(0);
        const xt::xtensor<double, 2> &x_g = mesh->geometry().x();
        std::vector<double> coordinate_dofs(3 * num_dofs_g);

        // TODO Get dofmap data
        // TODO Dof transformation stuff

        // FIXME Think of how to do this nicely. Should use integral_ids
        // FIXME TODO Permutations. See dolfinx exterior facet integrals
        // TODO Active cells. NOTE For the facet space, getting the active
        // cells in the usual way below actually gets the active facet
        // numbers, as the facets are treated as cells

        // FIXME What if this has no cell or facet kernel etc.?
        const auto &a00_cell_kernel =
            a[0][0]->kernel(dolfinx::fem::IntegralType::cell, -1);
        const auto &a00_facet_kernel =
            a[0][0]->kernel(dolfinx::fem::IntegralType::exterior_facet, -1);
        const auto &a01_kernel =
            a[0][1]->kernel(dolfinx::fem::IntegralType::exterior_facet, -1);
        const auto &a10_kernel =
            a[1][0]->kernel(dolfinx::fem::IntegralType::exterior_facet, -1);
        const auto &a11_kernel =
            a[1][1]->kernel(dolfinx::fem::IntegralType::cell, -1);

        // FIXME This is clumsy. Make helper array / function?
        // Dofmaps and block size for the a00 form
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap00_0 =
            a[0][0]->function_spaces().at(0)->dofmap()->list();
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap00_1 =
            a[0][0]->function_spaces().at(1)->dofmap()->list();
        const int bs00_0 = a[0][0]->function_spaces().at(0)->dofmap()->bs();
        const int bs00_1 = a[0][0]->function_spaces().at(1)->dofmap()->bs();
        const int num_dofs00_0 = bs00_0 * dofmap00_0.links(0).size();
        const int num_dofs00_1 = bs00_1 * dofmap00_1.links(0).size();

        // Dofmaps and block size for the a01 form
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap01_0 =
            a[0][1]->function_spaces().at(0)->dofmap()->list();
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap01_1 =
            a[0][1]->function_spaces().at(1)->dofmap()->list();
        const int bs01_0 = a[0][1]->function_spaces().at(0)->dofmap()->bs();
        const int bs01_1 = a[0][1]->function_spaces().at(1)->dofmap()->bs();
        const int num_dofs01_0 = bs01_0 * dofmap01_0.links(0).size();
        const int num_dofs01_1 = bs01_1 * dofmap01_1.links(0).size();

        // Dofmaps and block size for the a10 form
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap10_0 =
            a[1][0]->function_spaces().at(0)->dofmap()->list();
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap10_1 =
            a[1][0]->function_spaces().at(1)->dofmap()->list();
        const int bs10_0 = a[1][0]->function_spaces().at(0)->dofmap()->bs();
        const int bs10_1 = a[1][0]->function_spaces().at(1)->dofmap()->bs();
        const int num_dofs10_0 = bs10_0 * dofmap10_0.links(0).size();
        const int num_dofs10_1 = bs10_1 * dofmap10_1.links(0).size();

        // Dofmaps and block size for the a11 form
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap11_0 =
            a[1][1]->function_spaces().at(0)->dofmap()->list();
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap11_1 =
            a[1][1]->function_spaces().at(1)->dofmap()->list();
        const int bs11_0 = a[1][1]->function_spaces().at(0)->dofmap()->bs();
        const int bs11_1 = a[1][1]->function_spaces().at(1)->dofmap()->bs();
        const int num_dofs11_0 = bs11_0 * dofmap11_0.links(0).size();
        const int num_dofs11_1 = bs11_1 * dofmap11_1.links(0).size();

        // TODO Is this needed?
        mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
        auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
        assert(c_to_f);

        mesh->topology_mutable().create_entity_permutations();
        const std::vector<std::uint8_t> &perms =
            mesh->topology().get_facet_permutations();

        for (int c = 0; c < c_to_f->num_nodes(); ++c)
        {
            // Matrix to store the statically condensed system
            auto cell_facets = c_to_f->links(c);
            const int Ae_sc_num_rows = cell_facets.size() * num_dofs11_0;
            const int Ae_sc_num_cols = cell_facets.size() * num_dofs11_1;
            xt::xarray<double> Ae_sc = xt::zeros<double>({Ae_sc_num_rows,
                                                          Ae_sc_num_cols});

            std::cout << "a[0][0] = \n"
                      << assemble_cell_matrix(
                             *a[0][0], c, cell_facets,
                             constants, coeffs)
                      << "\na[0][1] = \n"
                      << assemble_cell_matrix(
                             *a[0][1], c, cell_facets,
                             constants, coeffs)
                      << "\na[1][0] = \n"
                      << assemble_cell_matrix(
                             *a[1][0], c, cell_facets,
                             constants, coeffs)
                      << "\na[1][1] = \n"
                      << assemble_cell_matrix(
                             *a[1][1], c, cell_facets,
                             constants, coeffs)
                      << "\n";

            xt::xarray<double> Ae00 = xt::zeros<double>({num_dofs00_0,
                                                         num_dofs00_1});

            // FIXME Why can't I put this in {}?
            const int Ae01_num_cols = cell_facets.size() * num_dofs01_1;
            xt::xarray<double> Ae01 =
                xt::zeros<double>({num_dofs01_0, Ae01_num_cols});

            const int Ae10_num_rows = cell_facets.size() * num_dofs10_0;
            xt::xarray<double> Ae10 =
                xt::zeros<double>({Ae10_num_rows, num_dofs10_1});

            // Get cell coordinates/geometry
            auto x_dofs = x_dofmap.links(c);
            for (std::size_t i = 0; i < x_dofs.size(); ++i)
            {
                std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                            std::next(coordinate_dofs.begin(), 3 * i));
            }

            // FIXME Passing cell to coeffs that were created for facets.
            // FIXME Check that Ae00.data() for an xtensor array is correct here
            // NOTE These FIXMEs apply to other kernel calls too
            a00_cell_kernel(Ae00.data(), coeffs.row(c).data(), constants.data(),
                            coordinate_dofs.data(), nullptr, nullptr);

            for (auto f : cell_facets)
            {
                // Get local index of facet with respect to the cell
                // TODO Could just loop through local facets or do this
                // some other way
                auto it = std::find(cell_facets.begin(), cell_facets.end(), f);
                assert(it != cell_facets.end());
                const int local_f = std::distance(cell_facets.begin(), it);

                // TODO Permutatations. This applies to other kernel calls
                a00_facet_kernel(Ae00.data(), coeffs.row(c).data(), constants.data(),
                                 coordinate_dofs.data(), &local_f,
                                 &perms[c * cell_facets.size() + local_f]);

                xt::xarray<double> Ae01_f = xt::zeros<double>({num_dofs01_0,
                                                               num_dofs01_1});

                a01_kernel(Ae01_f.data(), coeffs.row(c).data(), constants.data(),
                           coordinate_dofs.data(), &local_f,
                           &perms[c * cell_facets.size() + local_f]);

                // NOTE: For loop loops through global facet nums, but I think
                // this is in order of ascending local facet num. Check though
                // before relying on this.
                const int start_col = local_f * num_dofs01_1;
                const int end_col = start_col + num_dofs01_1;
                xt::view(Ae01, xt::all(), xt::range(start_col, end_col)) = Ae01_f;

                xt::xarray<double> Ae10_f = xt::zeros<double>({num_dofs10_0,
                                                               num_dofs10_1});
                a10_kernel(Ae10_f.data(), coeffs.row(c).data(), constants.data(),
                           coordinate_dofs.data(), &local_f,
                           &perms[c * cell_facets.size() + local_f]);

                const int start_row = local_f * num_dofs10_0;
                const int end_row = start_row + num_dofs10_0;
                xt::view(Ae10, xt::range(start_row, end_row), xt::all()) = Ae10_f;

                // FIXME Get facet coord dofs properly and check
                auto facet_x_dofs = mesh->topology().connectivity(tdim - 1, tdim - 2)->links(f);
                std::vector<double> fact_coordinate_dofs(3 * mesh->topology().connectivity(tdim - 1, tdim - 2)->num_links(0));
                for (std::size_t i = 0; i < facet_x_dofs.size(); ++i)
                {
                    std::copy_n(xt::row(x_g, facet_x_dofs[i]).begin(), 3,
                                std::next(fact_coordinate_dofs.begin(), 3 * i));
                }

                xt::xarray<double> Ae11_f = xt::zeros<double>({num_dofs11_0,
                                                               num_dofs11_1});
                a11_kernel(Ae11_f.data(), coeffs.row(c).data(), constants.data(),
                           fact_coordinate_dofs.data(), nullptr, nullptr);

                // NOTE Could compute from num_dofs11_* but the correct index
                // can be found from start_row / start_col etc.
                // FIXME/TODO num_dofs10_0 will be same as num_dofs11_0 (though
                // num_dofs10_1 will not be the same as num_dofs11_1). Simplify
                // using this fact
                xt::view(Ae_sc,
                         xt::range(start_row, end_row),
                         xt::range(start_col, end_col)) = Ae11_f;
            }

            // NOTE: xt::linalg::dot does matrix-vector and matrix matrix multiplication
            Ae_sc -= xt::linalg::dot(Ae10, xt::linalg::solve(Ae00, Ae01));

            // TODO Dirichlet BCs

            // Double loop over cell facets to assemble
            // FIXME Is there a better way?
            // TODO Here I use cell_facets[local_facet] = global_facet. Double check and
            // see if this is better for above or when making sparsity pattern!
            for (int local_f_i = 0; local_f_i < cell_facets.size(); ++local_f_i)
            {
                for (int local_f_j = 0; local_f_j < cell_facets.size(); ++local_f_j)
                {

                    const int f_i = cell_facets[local_f_i];
                    const int f_j = cell_facets[local_f_j];

                    auto dofs0 = dofmap11_0.links(f_i);
                    auto dofs1 = dofmap11_1.links(f_j);

                    // Matrix corresponding to dofs of facets f_i and f_j
                    // NOTE Have to cast to xt::xarray<double> (can't just use auto)
                    // otherwise it returns a view and Ae_sc_f_ij.data() gets the
                    // wrong values (the values) from the full Ae_sc array
                    xt::xarray<double> Ae_sc_f_ij =
                        xt::view(Ae_sc,
                                 xt::range(local_f_i * num_dofs11_0,
                                           local_f_i * num_dofs11_0 + num_dofs11_0),
                                 xt::range(local_f_j * num_dofs11_1,
                                           local_f_j * num_dofs11_1 + num_dofs11_1));

                    // FIXME Might be better/more efficient to do this to Ae_sc
                    // directly.
                    if (!bc0.empty())
                    {
                        for (int i = 0; i < num_dofs11_0; ++i)
                        {
                            for (int k = 0; k < bs11_0; ++k)
                            {
                                if (bc0[bs11_0 * dofs0[i] + k])
                                {
                                    // Zero row bs0 * i + k
                                    const int row = bs11_0 * i + k;
                                    xt::view(Ae_sc_f_ij, row, xt::all()) = 0.0;
                                }
                            }
                        }
                    }

                    if (!bc1.empty())
                    {
                        for (int j = 0; j < num_dofs11_1; ++j)
                        {
                            for (int k = 0; k < bs11_1; ++k)
                            {
                                if (bc1[bs11_1 * dofs1[j] + k])
                                {
                                    // Zero column bs1 * j + k
                                    const int col = bs11_1 * j + k;
                                    xt::view(Ae_sc_f_ij, xt::all(), col) = 0.0;
                                }
                            }
                        }
                    }

                    // NOTE dofs0.size() is same as num_dofs11_0 etc.
                    mat_set(dofs0.size(), dofs0.data(),
                            dofs1.size(), dofs1.data(),
                            Ae_sc_f_ij.data());
                }
            }
        }
    }
}
