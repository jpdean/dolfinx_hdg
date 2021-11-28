#pragma once

#include <dolfinx/fem/Form.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/mesh/Topology.h>

namespace dolfinx_hdg::fem
{
    dolfinx::la::SparsityPattern create_sparsity_pattern(
        const dolfinx::mesh::Topology &topology,
        const std::array<const std::reference_wrapper<const dolfinx::fem::DofMap>, 2> &
            dofmaps);

    template <typename T>
    dolfinx::la::SparsityPattern create_sparsity_pattern(
        const dolfinx::fem::Form<T> &a)
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
        // TODO Is this needed?
        mesh->topology_mutable().create_entities(tdim - 1);
        mesh->topology_mutable().create_connectivity(tdim, tdim - 1);

        return create_sparsity_pattern(mesh->topology(), dofmaps);
    }

    template <typename T>
    void pack(const std::uint32_t entity, int bs, const xtl::span<T> &entity_coeff,
              const xtl::span<const T> &v,
              const xtl::span<const std::uint32_t> &cell_info,
              const dolfinx::fem::DofMap &dofmap,
              const std::function<void(const xtl::span<T> &,
                                       const xtl::span<const std::uint32_t> &,
                                       std::int32_t, int)> &transform)
    {
        auto dofs = dofmap.cell_dofs(entity);
        for (std::size_t i = 0; i < dofs.size(); ++i)
        {
            std::copy_n(std::next(v.begin(), bs * dofs[i]), bs,
                        std::next(entity_coeff.begin(), bs * i));
        }

        // FIXME / TODO Need to use apply dof transformations but this would need
        // cell info where the cell is a facet. 
        // transform(cell_coeff, cell_info, cell, 1);
    }

    template <typename T, typename E, typename Functor>
    void pack_coefficient_entity(
        const xtl::span<T> &c, const int cstride,
        const int facet_cstride,
        const xtl::span<const T> &v,
        const xtl::span<const std::uint32_t> &cell_info,
        const dolfinx::fem::DofMap &dofmap,
        const xtl::span<const E> &active_entities, Functor fetch_cells,
        std::int32_t offset, int space_dim,
        const std::function<void(const xtl::span<T> &,
                                 const xtl::span<const std::uint32_t> &,
                                 std::int32_t, int)> &transformation,
        const dolfinx::graph::AdjacencyList<std::int32_t>& c_to_f)
    {
        const int bs = dofmap.bs();
        for (std::size_t e = 0; e < active_entities.size(); ++e)
        {
            std::int32_t cell = fetch_cells(active_entities[e]);

            auto facets = c_to_f.links(cell);

            for (int i = 0; i < facets.size(); ++i)
            {
                auto facet_coeff = c.subspan(e * cstride + i * facet_cstride + offset,
                                             space_dim);
                pack(facets[i], bs, facet_coeff, v, cell_info, dofmap, transformation);
            }
        }
    }

    template <typename T>
    std::pair<std::vector<T>, int>
    pack_coefficients(const dolfinx::fem::Form<T> &u,
                      dolfinx::fem::IntegralType integral_type, int id)
    {
        // Get form coefficient offsets and dofmaps
        const std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> coefficients = u.coefficients();
        const std::vector<int> offsets = u.coefficient_offsets();
        std::vector<const dolfinx::fem::DofMap *> dofmaps(coefficients.size());
        std::vector<const dolfinx::fem::FiniteElement *> elements(coefficients.size());
        std::vector<xtl::span<const T>> v;
        v.reserve(coefficients.size());
        for (std::size_t i = 0; i < coefficients.size(); ++i)
        {
            elements[i] = coefficients[i]->function_space()->element().get();
            dofmaps[i] = coefficients[i]->function_space()->dofmap().get();
            v.push_back(coefficients[i]->x()->array());
        }

        // Get mesh
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = u.mesh();
        assert(mesh);

        const int tdim = mesh->topology().dim();
        mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
        auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
        assert(c_to_f);

        // Copy data into coefficient array
        const int num_cell_facets
            = dolfinx::mesh::cell_num_entities(mesh->topology().cell_type(),
                                               tdim - 1);
        // Pack coefficients cellwise but for each facet of that cell
        const int cstride = offsets.back() * num_cell_facets;
        const int facet_cstride = offsets.back();
        std::vector<T> c;
        if (!coefficients.empty())
        {
            bool needs_dof_transformations = false;
            for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
            {
                if (elements[coeff]->needs_dof_transformations())
                {
                    needs_dof_transformations = true;
                    mesh->topology_mutable().create_entity_permutations();
                }
            }

            xtl::span<const std::uint32_t> cell_info;
            if (needs_dof_transformations)
                cell_info = xtl::span(mesh->topology().get_cell_permutation_info());

            // TODO see if this can be simplified with templating
            switch (integral_type)
            {
            case dolfinx::fem::IntegralType::cell:
            {
                const std::vector<std::int32_t> &active_cells = u.cell_domains(id);
                c.resize(active_cells.size() * num_cell_facets * offsets.back());

                // Iterate over coefficients
                for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
                {
                    const auto transform = elements[coeff]->get_dof_transformation_function<T>(false, true);
                    pack_coefficient_entity<T, std::int32_t>(
                        xtl::span<T>(c), cstride, facet_cstride, v[coeff], cell_info, *dofmaps[coeff],
                        active_cells, [](std::int32_t entity)
                        { return entity; },
                        offsets[coeff], elements[coeff]->space_dimension(), transform,
                        *c_to_f);
                }
                break;
            }
            default:
                throw std::runtime_error(
                    "Could not pack coefficient. Integral type not supported.");
            }
        }
        return {std::move(c), cstride};
    }

    template <typename T>
    std::map<std::pair<dolfinx::fem::IntegralType, int>,
             std::pair<std::vector<T>, int>>
    pack_coefficients(const dolfinx::fem::Form<T> &u)
    {
        std::map<std::pair<dolfinx::fem::IntegralType, int>,
                 std::pair<std::vector<T>, int>>
            coefficients;

        for (auto integral_type : u.integral_types())
        {
            for (int i : u.integral_ids(integral_type))
            {
                coefficients.emplace(std::pair(integral_type, i),
                               dolfinx_hdg::fem::pack_coefficients(u, integral_type, i));
            }
        }
        return coefficients;
    }
}
