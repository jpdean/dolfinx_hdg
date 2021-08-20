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
        const dolfinx::fem::Form<T> &a,
        const xtl::span<const T> &constants,
        const dolfinx::array2d<T> &coeffs, const std::vector<bool> &bc0,
        const std::vector<bool> &bc1)
    {
        // FIXME Vector elements (i.e. block size \neq 1) might break some of this
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a.mesh();
        assert(mesh);
        const int tdim = mesh->topology().dim();

        // FIXME Check
        const dolfinx::graph::AdjacencyList<std::int32_t> &x_dofmap =
            mesh->geometry().dofmap();
        const std::size_t num_dofs_g = x_dofmap.num_links(cell);
        const xt::xtensor<double, 2> &x_g = mesh->geometry().x();
        std::vector<double> coordinate_dofs(3 * num_dofs_g);

        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap0 =
            a.function_spaces().at(0)->dofmap()->list();
        const int bs0 = a.function_spaces().at(0)->dofmap()->bs();
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap1 =
            a.function_spaces().at(1)->dofmap()->list();
        const int bs1 = a.function_spaces().at(1)->dofmap()->bs();
        const int num_dofs0 = dofmap0.links(0).size();
        const int num_dofs1 = dofmap1.links(0).size();
        const int ndim0 = bs0 * num_dofs0;
        const int ndim1 = bs1 * num_dofs1;

        // TODO Is this needed?
        mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
        auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
        assert(c_to_f);
        mesh->topology_mutable().create_entity_permutations();
        const std::vector<std::uint8_t> &perms =
            mesh->topology().get_facet_permutations();

        // FIXME Do this properly
        const auto &kernel =
            a.kernel(dolfinx::fem::IntegralType::cell, -1);

        for (int c = 0; c < c_to_f->num_nodes(); ++c)
        {
            auto cell_facets = c_to_f->links(c);
            const int num_facets = cell_facets.size();

            // FIXME Check
            auto x_dofs = x_dofmap.links(c);
            for (std::size_t i = 0; i < x_dofs.size(); ++i)
            {
                std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                            std::next(coordinate_dofs.begin(), 3 * i));
            }

            // TODO Do this properly
            xt::xarray<double> Ae_sc = xt::zeros<double>({ndim0 * num_facets,
                                                          ndim1 * num_facets});

            kernel(Ae_sc.data(), coeffs.row(c).data(), constants.data(),
                   coordinate_dofs.data(), nullptr, nullptr);

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

                    auto dofs0 = dofmap0.links(f_i);
                    auto dofs1 = dofmap1.links(f_j);

                    // Matrix corresponding to dofs of facets f_i and f_j
                    // NOTE Have to cast to xt::xarray<double> (can't just use auto)
                    // otherwise it returns a view and Ae_sc_f_ij.data() gets the
                    // wrong values (the values) from the full Ae_sc array
                    xt::xarray<double> Ae_sc_f_ij =
                        xt::view(Ae_sc,
                                 xt::range(local_f_i * ndim0,
                                           local_f_i * ndim0 + ndim0),
                                 xt::range(local_f_j * ndim1,
                                           local_f_j * ndim1 + ndim1));

                    // FIXME Might be better/more efficient to do this to Ae_sc
                    // directly.
                    if (!bc0.empty())
                    {
                        for (int i = 0; i < num_dofs0; ++i)
                        {
                            for (int k = 0; k < bs0; ++k)
                            {
                                if (bc0[bs0 * dofs0[i] + k])
                                {
                                    // Zero row bs0 * i + k
                                    const int row = bs0 * i + k;
                                    xt::view(Ae_sc_f_ij, row, xt::all()) = 0.0;
                                }
                            }
                        }
                    }

                    if (!bc1.empty())
                    {
                        for (int j = 0; j < num_dofs1; ++j)
                        {
                            for (int k = 0; k < bs1; ++k)
                            {
                                if (bc1[bs1 * dofs1[j] + k])
                                {
                                    // Zero column bs1 * j + k
                                    const int col = bs1 * j + k;
                                    xt::view(Ae_sc_f_ij, xt::all(), col) = 0.0;
                                }
                            }
                        }
                    }

                    mat_set(dofs0.size(), dofs0.data(),
                            dofs1.size(), dofs1.data(),
                            Ae_sc_f_ij.data());
                }
            }
        }
    }
}
