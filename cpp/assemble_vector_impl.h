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
                         const std::vector<std::shared_ptr<
                             const dolfinx::fem::Form<PetscScalar>>> &L,
                         const std::vector<std::vector<std::shared_ptr<
                             const dolfinx::fem::Form<T>>>> &a,
                         const xtl::span<const T> &constants,
                         const dolfinx::array2d<T> &coeffs)
    {
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = L[1]->mesh();
        assert(mesh);
        const int tdim = mesh->topology().dim();

        assert(L[1]->function_spaces().at(0));
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap =
            L[1]->function_spaces().at(0)->dofmap()->list();
        assert(dofmap);
        const int bs = L[1]->function_spaces().at(0)->dofmap()->bs();
        const int num_dofs = dofmap.links(0).size();
        const int ndim = bs * num_dofs;

        // TODO DOF Transformations

        // TODO Is this needed?
        mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
        auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
        assert(c_to_f);
        mesh->topology_mutable().create_entity_permutations();
        const std::vector<std::uint8_t> &perms =
            mesh->topology().get_facet_permutations();

        for (int c = 0; c < c_to_f->num_nodes(); ++c)
        {
            auto cell_facets = c_to_f->links(c);

            auto be0 = dolfinx_hdg::sc::assemble_cell_vector(*L[0], c, cell_facets,
                                                             constants, coeffs);
            auto Ae10 = dolfinx_hdg::sc::assemble_cell_matrix(
                *a[1][0], c, cell_facets,
                constants, coeffs);
            auto Ae00 = dolfinx_hdg::sc::assemble_cell_matrix(
                *a[0][0], c, cell_facets,
                constants, coeffs);

            // Call "be1" be_sc as this is the starting point for the
            // statically condensed vector
            auto be_sc = dolfinx_hdg::sc::assemble_cell_vector(*L[1], c, cell_facets,
                                                               constants, coeffs);
            // NOTE: xt::linalg::dot does matrix-vector and matrix matrix
            // multiplication
            be_sc -= xt::linalg::dot(Ae10, xt::linalg::solve(Ae00, be0));

            for (int local_f = 0; local_f < cell_facets.size(); ++local_f)
            {
                const int f = cell_facets[local_f];

                auto dofs = dofmap.links(f);

                // Vector corresponding to dofs of facets f
                // NOTE Have to cast to xt::xarray<double> (can't just use auto)
                // otherwise it returns a view and Le_sc_f.data() gets the
                // wrong values (the values) from the full Ae_sc array
                xt::xarray<double> be_sc_f =
                    xt::view(be_sc,
                             xt::range(local_f * ndim,
                                       local_f * ndim + ndim));

                for (int i = 0; i < num_dofs; ++i)
                    for (int k = 0; k < bs; ++k)
                        b[bs * dofs[i] + k] += be_sc_f[bs * i + k];
            }
        }
    }
}
