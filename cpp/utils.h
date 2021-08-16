#pragma once

#include <dolfinx/fem/Form.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/mesh/Topology.h>

namespace dolfinx_hdg::fem
{
    dolfinx::la::SparsityPattern create_sparsity_pattern(
    const dolfinx::mesh::Topology& topology,
    const std::array<const std::reference_wrapper<const dolfinx::fem::DofMap>, 2>&
        dofmaps);

    template <typename T>
    dolfinx::la::SparsityPattern create_sparsity_pattern(
        const dolfinx::fem::Form<T>& a)
    {
        if (a.rank() != 2)
        {
            throw std::runtime_error(
                "Cannot create sparsity pattern. Form is not a bilinear form");
        }

        // Get dof maps and mesh
        std::array<const std::reference_wrapper<const dolfinx::fem::DofMap>, 2>
            dofmaps{
                *a.function_spaces().at(0)->dofmap(),
                *a.function_spaces().at(1)->dofmap()};
        std::shared_ptr mesh = a.mesh();
        assert(mesh);

        const int tdim = mesh->topology().dim();
        mesh->topology_mutable().create_entities(tdim - 1);
        // TODO Is this needed?
        // mesh->topology_mutable().create_connectivity(tdim - 1, tdim);

        return create_sparsity_pattern(mesh->topology(), dofmaps);
    }
}
