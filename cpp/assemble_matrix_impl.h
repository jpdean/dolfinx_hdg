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
    // FIXME Do I need the forward declaration?
    template <typename T>
    void assemble_matrix(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_set,
        const dolfinx::fem::Form<T> &a, const xtl::span<const T> &constants,
        const dolfinx::array2d<T> &coeffs, const std::vector<bool> &bc0,
        const std::vector<bool> &bc1);

    template <typename T>
    void assemble_cells(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_set,
        const dolfinx::mesh::Mesh &mesh,
        const std::function<void(const xtl::span<T> &,
                                 const xtl::span<const std::uint32_t> &,
                                 std::int32_t, int)>
            apply_dof_transformation,
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap0, const int bs0,
        const std::function<void(const xtl::span<T> &,
                                 const xtl::span<const std::uint32_t> &,
                                 std::int32_t, int)>
            apply_dof_transformation_to_transpose,
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap1, const int bs1,
        const std::vector<bool> &bc0, const std::vector<bool> &bc1,
        const std::function<void(T *, const T *, const T *, const double *, const int *,
                                 const std::uint8_t *)> &kernel,
        const dolfinx::array2d<T> &coeffs, const xtl::span<const T> &constants,
        const xtl::span<const std::uint32_t> &cell_info)
    {
        // Iterate over active cells
        const int num_dofs0 = dofmap0.links(0).size();
        const int num_dofs1 = dofmap1.links(0).size();
        const int ndim0 = bs0 * num_dofs0;
        const int ndim1 = bs1 * num_dofs1;
        std::vector<T> Ae(ndim0 * ndim1);

        const int tdim = mesh.topology().dim();
        auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
        assert(c_to_f);

        for (int c = 0; c < c_to_f->num_nodes(); ++c)
        {
            std::cout << "c = " << c << "\n";
            for (auto f : c_to_f->links(c))
            {
                std::cout << "  f = " << f << "\n";
            }
        }
    }

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

        // FIXME Think how to do this properly
        for (int i : a.integral_ids(dolfinx::fem::IntegralType::cell))
        {
            // TODO Permutations
            const auto &fn = a.kernel(dolfinx::fem::IntegralType::cell, i);
            // TODO Active cells. NOTE For the facet space, getting the active
            // cells in the usual way below actually gets the active facet
            // numbers, as the facets are treated as cells
            // const std::vector<std::int32_t> &active_cells =
            //     a.domains(dolfinx::fem::IntegralType::cell, i);
            // TODO Is this needed?
            mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
            assemble_cells<T>(mat_set, *mesh,
                              apply_dof_transformation, dofs0, bs0,
                              apply_dof_transformation_to_transpose, dofs1, bs1,
                              bc0, bc1, fn, coeffs, constants, cell_info);
        }
    }
}
