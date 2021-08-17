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
            std::cout << "c = " << c << "\n";

            // Matrix to store the statically condensed system
            auto cell_facets = c_to_f->links(c);
            const int Ae_sc_num_rows = cell_facets.size() * num_dofs11_0;
            const int Ae_sc_num_cols = cell_facets.size() * num_dofs11_1;
            xt::xarray<double> Ae_sc = xt::zeros<double>({Ae_sc_num_rows,
                                                          Ae_sc_num_cols});

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
            std::cout << Ae_sc << "\n";

            // TODO Dirichlet BCs
        }
    }
}
