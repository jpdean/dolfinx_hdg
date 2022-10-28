#pragma once

#include <dolfinx/fem/Form.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/Mesh.h>

namespace dolfinx_hdg::fem
{
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
        std::array<std::reference_wrapper<const dolfinx::fem::DofMap>, 2> dofmaps{
            *a.function_spaces().at(0)->dofmap(),
            *a.function_spaces().at(1)->dofmap()};
        std::shared_ptr mesh = a.mesh();
        assert(mesh);

        // FIXME: cleanup these calls? Some of the happen internally again.
        const int tdim = mesh->topology().dim();
        mesh->topology_mutable().create_entities(tdim - 1);
        mesh->topology_mutable().create_connectivity(tdim, tdim - 1);

        // Get common::IndexMaps for each dimension
        const std::array index_maps{dofmaps[0].get().index_map,
                                    dofmaps[1].get().index_map};
        const std::array bs = {dofmaps[0].get().index_map_bs(),
                               dofmaps[1].get().index_map_bs()};

        dolfinx::la::SparsityPattern pattern(mesh->comm(), index_maps, bs);

        const auto entity_map_0 = a.function_space_to_entity_map(*a.function_spaces().at(0));
        const auto entity_map_1 = a.function_space_to_entity_map(*a.function_spaces().at(1));

        const int num_cell_facets =
            dolfinx::mesh::cell_num_entities(mesh->topology().cell_type(), tdim - 1);

        std::vector<int> ids = a.integral_ids(dolfinx::fem::IntegralType::cell);
        for (int id : ids)
        {
            const std::vector<std::int32_t> &cells = a.cell_domains(id);
            // TODO Create sparsity

            for (std::int32_t cell : cells)
            {
                for (int local_facet_0 = 0; local_facet_0 < num_cell_facets; ++local_facet_0)
                {
                    // FIXME Tidy
                    const std::array cell_local_facet_0 = {cell, local_facet_0};
                    const std::int32_t facet_0 = entity_map_0(cell_local_facet_0);

                    for (int local_facet_1 = 0; local_facet_1 < num_cell_facets; ++local_facet_1)
                    {
                        const std::array cell_local_facet_1 = {cell, local_facet_1};
                        const std::int32_t facet_1 = entity_map_1(cell_local_facet_1);

                        pattern.insert(
                            dofmaps[0].get().cell_dofs(facet_0),
                            dofmaps[1].get().cell_dofs(facet_1));
                    }
                }
            }
        }

        return pattern;
    }

    // /// @brief Pack coefficients of a Form for a given integral type and
    // /// domain id
    // /// @param[in] form The Form
    // /// @param[in] integral_type Type of integral
    // /// @param[in] id The id of the integration domain
    // /// @param[in] c The coefficient array
    // /// @param[in] cstride The coefficient stride
    // template <typename T>
    // void pack_coefficients(const Form<T>& form, IntegralType integral_type, int id,
    //                    const std::span<T>& c, int cstride)
    // {
    //     // Get form coefficient offsets and dofmaps
    //     const std::vector<std::shared_ptr<const Function<T>>> &coefficients = form.coefficients();
    //     const std::vector<int> offsets = form.coefficient_offsets();

    //     if (!coefficients.empty())
    //     {
    //         switch (integral_type)
    //         {
    //         case IntegralType::cell:
    //         {
    //             const std::vector<std::int32_t> &cells = form.cell_domains(id);
    //             // Iterate over coefficients
    //             for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
    //             {
    //                 // Other integrals in the form might have coefficients defined over
    //                 // entities of codim > 0, which don't make sense for cell integrals, so
    //                 // don't pack them.
    //                 const int tdim = form.mesh()->topology().dim();
    //                 const int codim = tdim - coefficients[coeff]->function_space()->mesh()->topology().dim();
    //                 if (codim != 0)
    //                     continue;

    //                 auto fetch_cell = form.function_space_to_entity_map(
    //                     *coefficients[coeff]->function_space());
    //                 // Get cell info for coefficient (with respect to coefficient mesh)
    //                 std::span<const std::uint32_t> cell_info = impl::get_cell_orientation_info(*coefficients[coeff]);
    //                 impl::pack_coefficient_entity(c, cstride, *coefficients[coeff],
    //                                               cell_info, cells, 1, fetch_cell,
    //                                               offsets[coeff]);
    //             }
    //             break;
    //         }
    //         case IntegralType::exterior_facet:
    //         {
    //             const std::vector<std::int32_t> &facets = form.exterior_facet_domains(id);

    //             // Iterate over coefficients
    //             for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
    //             {
    //                 // Create lambda function fetching cell index from exterior facet entity
    //                 auto fetch_cell = form.function_space_to_entity_map(
    //                     *coefficients[coeff]->function_space());
    //                 std::span<const std::uint32_t> cell_info = impl::get_cell_orientation_info(*coefficients[coeff]);
    //                 impl::pack_coefficient_entity(c, cstride, *coefficients[coeff],
    //                                               cell_info, facets, 2, fetch_cell,
    //                                               offsets[coeff]);
    //             }

    //             break;
    //         }
    //         case IntegralType::interior_facet:
    //         {
    //             const std::vector<std::int32_t> &facets = form.interior_facet_domains(id);

    //             // Iterate over coefficients
    //             for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
    //             {
    //                 auto entity_map = form.function_space_to_entity_map(
    //                     *coefficients[coeff]->function_space());
    //                 // Lambda functions to fetch cell index from interior facet entity
    //                 auto fetch_cell0 = [entity_map](auto &entity)
    //                 { return entity_map(entity.subspan(0, 2)); };
    //                 auto fetch_cell1 = [entity_map](auto &entity)
    //                 { return entity_map(entity.subspan(2, 4)); };

    //                 std::span<const std::uint32_t> cell_info = impl::get_cell_orientation_info(*coefficients[coeff]);
    //                 // Pack coefficient ['+']
    //                 impl::pack_coefficient_entity(c, 2 * cstride, *coefficients[coeff],
    //                                               cell_info, facets, 4, fetch_cell0,
    //                                               2 * offsets[coeff]);
    //                 // Pack coefficient ['-']
    //                 impl::pack_coefficient_entity(c, 2 * cstride, *coefficients[coeff],
    //                                               cell_info, facets, 4, fetch_cell1,
    //                                               offsets[coeff] + offsets[coeff + 1]);
    //             }
    //             break;
    //         }
    //         default:
    //             throw std::runtime_error(
    //                 "Could not pack coefficient. Integral type not supported.");
    //         }
    //     }
    // }

    /// @brief Allocate storage for coefficients of a pair (integral_type,
    /// id) from a fem::Form form
    /// @param[in] form The Form
    /// @param[in] integral_type Type of integral
    /// @param[in] id The id of the integration domain
    /// @return A storage container and the column stride
    template <typename T>
    std::pair<std::vector<T>, int>
    allocate_coefficient_storage(const dolfinx::fem::Form<T> &form,
                                 dolfinx::fem::IntegralType integral_type,
                                 int id)
    {
        // Get form coefficient offsets and dofmaps
        const std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> &coefficients = form.coefficients();
        const std::vector<int> offsets = form.coefficient_offsets();

        // TODO This needs to be generalised for non-facet space coeffs
        std::size_t num_entities = 0;
        int cstride = 0;
        if (!coefficients.empty())
        {
            cstride = offsets.back();
            if (integral_type != dolfinx::fem::IntegralType::cell)
                throw std::runtime_error(
                    "Could not pack coefficient. Integral type not supported.");

            std::size_t num_cells = form.cell_domains(id).size();
            std::shared_ptr<const mesh::Mesh> mesh = form.mesh();
            assert(mesh);
            const int num_cell_facets = mesh::cell_num_entities(mesh->topology().cell_type(), mesh->topology().dim() - 1);
            num_entities = num_cell_facets * num_cells;
        }

        return {std::vector<T>(num_entities * cstride), cstride};
    }

    /// @brief Allocate memory for packed coefficients of a Form
    /// @param[in] form The Form
    /// @return A map from a form (integral_type, domain_id) pair to a
    /// (coeffs, cstride) pair
    template <typename T>
    std::map<std::pair<dolfinx::fem::IntegralType, int>, std::pair<std::vector<T>, int>>
    allocate_coefficient_storage(const dolfinx::fem::Form<T> &form)
    {
        std::map<std::pair<dolfinx::fem::IntegralType, int>,
                 std::pair<std::vector<T>, int>>
            coeffs;
        for (auto integral_type : form.integral_types())
        {
            for (int id : form.integral_ids(integral_type))
            {
                coeffs.emplace_hint(
                    coeffs.end(), std::pair(integral_type, id),
                    dolfinx_hdg::fem::allocate_coefficient_storage(form, integral_type, id));
            }
        }

        return coeffs;
    }

    /// @brief Pack coefficients of a Form
    /// @param[in] form The Form
    /// @param[in] coeffs A map from a (integral_type, domain_id) pair to a
    /// (coeffs, cstride) pair
    template <typename T>
    void pack_coefficients(const dolfinx::fem::Form<T> &form,
                           std::map<std::pair<dolfinx::fem::IntegralType, int>,
                                    std::pair<std::vector<T>, int>> &coeffs)
    {
        for (auto &[key, val] : coeffs)
        {
            if (key.first != dolfinx::fem::IntegralType::cell)
                throw std::runtime_error(
                    "Could not pack coefficient. Integral type not supported.");

            // pack_coefficients<T>(form, key.second, val.first, val.second);
            std::cout << "TODO Pack coeffs\n";
        }
    }

    // template <typename T>
    // void pack(const std::uint32_t entity, int bs, const xtl::span<T> &entity_coeff,
    //           const xtl::span<const T> &v,
    //           const xtl::span<const std::uint32_t> &cell_info,
    //           const dolfinx::fem::DofMap &dofmap,
    //           const std::function<void(const xtl::span<T> &,
    //                                    const xtl::span<const std::uint32_t> &,
    //                                    std::int32_t, int)> &transform)
    // {
    //     auto dofs = dofmap.cell_dofs(entity);
    //     for (std::size_t i = 0; i < dofs.size(); ++i)
    //     {
    //         std::copy_n(std::next(v.begin(), bs * dofs[i]), bs,
    //                     std::next(entity_coeff.begin(), bs * i));
    //     }

    //     // FIXME / TODO Need to use apply dof transformations but this would need
    //     // cell info where the cell is a facet.
    //     // transform(cell_coeff, cell_info, cell, 1);
    // }

    // template <typename T, typename E, typename Functor>
    // void pack_coefficient_entity(
    //     const xtl::span<T> &c, const int cstride,
    //     const int facet_cstride,
    //     const xtl::span<const T> &v,
    //     const xtl::span<const std::uint32_t> &cell_info,
    //     const dolfinx::fem::DofMap &dofmap,
    //     const xtl::span<const E> &active_entities, Functor fetch_cells,
    //     std::int32_t offset, int space_dim,
    //     const std::function<void(const xtl::span<T> &,
    //                              const xtl::span<const std::uint32_t> &,
    //                              std::int32_t, int)> &transformation,
    //     const dolfinx::graph::AdjacencyList<std::int32_t>& c_to_f)
    // {
    //     const int bs = dofmap.bs();
    //     for (std::size_t e = 0; e < active_entities.size(); ++e)
    //     {
    //         std::int32_t cell = fetch_cells(active_entities[e]);

    //         auto facets = c_to_f.links(cell);

    //         for (int i = 0; i < facets.size(); ++i)
    //         {
    //             auto facet_coeff = c.subspan(e * cstride + i * facet_cstride + offset,
    //                                          space_dim);
    //             pack(facets[i], bs, facet_coeff, v, cell_info, dofmap, transformation);
    //         }
    //     }
    // }

    // template <typename T>
    // std::pair<std::vector<T>, int>
    // pack_coefficients(const dolfinx::fem::Form<T> &u,
    //                   dolfinx::fem::IntegralType integral_type, int id)
    // {
    //     // Get form coefficient offsets and dofmaps
    //     const std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> coefficients = u.coefficients();
    //     const std::vector<int> offsets = u.coefficient_offsets();
    //     std::vector<const dolfinx::fem::DofMap *> dofmaps(coefficients.size());
    //     std::vector<const dolfinx::fem::FiniteElement *> elements(coefficients.size());
    //     std::vector<xtl::span<const T>> v;
    //     v.reserve(coefficients.size());
    //     for (std::size_t i = 0; i < coefficients.size(); ++i)
    //     {
    //         elements[i] = coefficients[i]->function_space()->element().get();
    //         dofmaps[i] = coefficients[i]->function_space()->dofmap().get();
    //         v.push_back(coefficients[i]->x()->array());
    //     }

    //     // Get mesh
    //     std::shared_ptr<const dolfinx::mesh::Mesh> mesh = u.mesh();
    //     assert(mesh);

    //     const int tdim = mesh->topology().dim();
    //     mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
    //     auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
    //     assert(c_to_f);

    //     // Copy data into coefficient array
    //     const int num_cell_facets
    //         = dolfinx::mesh::cell_num_entities(mesh->topology().cell_type(),
    //                                            tdim - 1);
    //     // Pack coefficients cellwise but for each facet of that cell
    //     const int cstride = offsets.back() * num_cell_facets;
    //     const int facet_cstride = offsets.back();
    //     std::vector<T> c;
    //     if (!coefficients.empty())
    //     {
    //         bool needs_dof_transformations = false;
    //         for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
    //         {
    //             if (elements[coeff]->needs_dof_transformations())
    //             {
    //                 needs_dof_transformations = true;
    //                 mesh->topology_mutable().create_entity_permutations();
    //             }
    //         }

    //         xtl::span<const std::uint32_t> cell_info;
    //         if (needs_dof_transformations)
    //             cell_info = xtl::span(mesh->topology().get_cell_permutation_info());

    //         // TODO see if this can be simplified with templating
    //         switch (integral_type)
    //         {
    //         case dolfinx::fem::IntegralType::cell:
    //         {
    //             const std::vector<std::int32_t> &active_cells = u.cell_domains(id);
    //             c.resize(active_cells.size() * num_cell_facets * offsets.back());

    //             // Iterate over coefficients
    //             for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
    //             {
    //                 const auto transform = elements[coeff]->get_dof_transformation_function<T>(false, true);
    //                 pack_coefficient_entity<T, std::int32_t>(
    //                     xtl::span<T>(c), cstride, facet_cstride, v[coeff], cell_info, *dofmaps[coeff],
    //                     active_cells, [](std::int32_t entity)
    //                     { return entity; },
    //                     offsets[coeff], elements[coeff]->space_dimension(), transform,
    //                     *c_to_f);
    //             }
    //             break;
    //         }
    //         default:
    //             throw std::runtime_error(
    //                 "Could not pack coefficient. Integral type not supported.");
    //         }
    //     }
    //     return {std::move(c), cstride};
    // }

    // template <typename T>
    // std::map<std::pair<dolfinx::fem::IntegralType, int>,
    //          std::pair<std::vector<T>, int>>
    // pack_coefficients(const dolfinx::fem::Form<T> &u)
    // {
    //     std::map<std::pair<dolfinx::fem::IntegralType, int>,
    //              std::pair<std::vector<T>, int>>
    //         coefficients;

    //     for (auto integral_type : u.integral_types())
    //     {
    //         for (int i : u.integral_ids(integral_type))
    //         {
    //             coefficients.emplace(std::pair(integral_type, i),
    //                            dolfinx_hdg::fem::pack_coefficients(u, integral_type, i));
    //         }
    //     }
    //     return coefficients;
    // }
}
