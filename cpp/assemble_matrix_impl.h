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

namespace dolfinx_hdg::fem::impl
{
    template <typename T>
    void assemble_matrix(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_set,
        const dolfinx::fem::Form<T> &a, const xtl::span<const T> &constants,
        const dolfinx::array2d<T> &coeffs, const std::vector<bool> &bc0,
        const std::vector<bool> &bc1);

    template <typename T>
    void assemble_matrix(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_set,
        const dolfinx::fem::Form<T> &a, const xtl::span<const T> &constants,
        const dolfinx::array2d<T> &coeffs, const std::vector<bool> &bc0,
        const std::vector<bool> &bc1)
    {
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a.mesh();
        assert(mesh);
        const int tdim = mesh->topology().dim();

        // Get dofmap data
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = a.function_spaces().at(0)->dofmap();
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = a.function_spaces().at(1)->dofmap();
        assert(dofmap0);
        assert(dofmap1);
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofs0 = dofmap0->list();
        const int bs0 = dofmap0->bs();
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofs1 = dofmap1->list();
        const int bs1 = dofmap1->bs();

        std::shared_ptr<const dolfinx::fem::FiniteElement> element0 = a.function_spaces().at(0)->element();
        std::shared_ptr<const dolfinx::fem::FiniteElement> element1 = a.function_spaces().at(1)->element();
        const std::function<void(const xtl::span<T> &,
                                 const xtl::span<const std::uint32_t> &, std::int32_t,
                                 int)>
            apply_dof_transformation = element0->get_dof_transformation_function<T>();
        const std::function<void(const xtl::span<T> &,
                                 const xtl::span<const std::uint32_t> &, std::int32_t,
                                 int)>
            apply_dof_transformation_to_transpose = element1->get_dof_transformation_to_transpose_function<T>();

        const bool needs_transformation_data = element0->needs_dof_transformations() or element1->needs_dof_transformations() or a.needs_facet_permutations();
        xtl::span<const std::uint32_t> cell_info;
        if (needs_transformation_data)
        {
            mesh->topology_mutable().create_entity_permutations();
            cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
        }
    }
}
