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

            auto b = dolfinx_hdg::sc::assemble_cell_vector(*L[0], c, cell_facets,
                                                           constants, coeffs);
            std::cout << b << "\n";
        }
    }
}
