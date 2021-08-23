#pragma once

// FIXME See which of these aren't needed
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <iostream>
#include "static_condensation_helpers.h"

namespace dolfinx_hdg::fem::impl
{
    template <typename T>
    void assemble_vector(xtl::span<T> b,
                         const dolfinx::fem::Form<PetscScalar> &L,
                         const xtl::span<const T> &constants,
                         const dolfinx::array2d<T> &coeffs)
    {
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = L.mesh();
        assert(mesh);
        const int tdim = mesh->topology().dim();

        assert(L.function_spaces().at(0));
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap =
            L.function_spaces().at(0)->dofmap()->list();
        assert(dofmap);
        const int bs = L.function_spaces().at(0)->dofmap()->bs();
        const int num_dofs = dofmap.links(0).size();
        const int ndim = bs * num_dofs;

        const int codim = L.function_spaces().at(0)->codimension();

        // TODO DOF Transformations

        // TODO Is this needed?
        mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
        auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
        assert(c_to_f);
        mesh->topology_mutable().create_entity_permutations();
        const std::vector<std::uint8_t> &perms =
            mesh->topology().get_facet_permutations();

        // FIXME Do this properly
        const auto &kernel =
            L.kernel(dolfinx::fem::IntegralType::cell, -1);

        for (int c = 0; c < c_to_f->num_nodes(); ++c)
        {
            auto cell_facets = c_to_f->links(c);
            const int num_facets = cell_facets.size();

            auto coordinate_dofs = dolfinx_hdg::sc::get_cell_coord_dofs(mesh, c);

            // TODO Do this properly
            // Check codim either 0 or 1
            const int be_size = codim == 0 ? ndim : ndim * num_facets;
            xt::xarray<double> be = xt::zeros<double>({be_size});

            std::vector<unsigned char> cell_facet_perms(num_facets);
            // FIXME This assumes codim = 1. Need to generalize.
            std::vector<T> cell_coeffs;
            for (int local_f = 0; local_f < num_facets; ++local_f)
            {
                cell_facet_perms[local_f] =
                    perms[c * num_facets + local_f];

                // Pack coefficients for each of the cells facets into one
                // array
                // FIXME Do this without loop
                for (auto coeff : coeffs.row(cell_facets[local_f]))
                {
                    cell_coeffs.push_back(coeff);
                }
            }

            kernel(be.data(), cell_coeffs.data(), constants.data(),
                   coordinate_dofs.data(), nullptr,
                   cell_facet_perms.data());

            if (codim == 0)
            {
                auto dofs = dofmap.links(c);
                for (int i = 0; i < num_dofs; ++i)
                    for (int k = 0; k < bs; ++k)
                        b[bs * dofs[i] + k] += be[bs * i + k];
            }
            else
            {
                for (int local_f = 0; local_f < cell_facets.size(); ++local_f)
                {
                    const int f = cell_facets[local_f];

                    auto dofs = dofmap.links(f);

                    // Vector corresponding to dofs of facets f
                    // NOTE Have to cast to xt::xarray<double> (can't just use auto)
                    // otherwise it returns a view and Le_sc_f.data() gets the
                    // wrong values (the values) from the full Ae_sc array
                    xt::xarray<double> be_sc_f =
                        xt::view(be,
                                 xt::range(local_f * ndim,
                                           local_f * ndim + ndim));

                    for (int i = 0; i < num_dofs; ++i)
                        for (int k = 0; k < bs; ++k)
                            b[bs * dofs[i] + k] += be_sc_f[bs * i + k];
                }
            }
        }
    }
}
