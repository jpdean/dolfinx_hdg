#pragma once

#include <dolfinx/fem/Form.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/Mesh.h>
#include <array>
#include <functional>
#include <iterator>
#include <vector>
#include <span>
#include <iostream>

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

    /// @brief Pack a single coefficient for a set of active entities
    ///
    /// @param[out] c The coefficient to be packed
    /// @param[in] cstride The total number of coefficient values to pack
    /// for each entity
    /// @param[in] u The function to extract data from
    /// @param[in] cell_info Array of bytes describing which transformation
    /// has to be applied on the cell to map it to the reference element
    /// @param[in] entities The set of active entities
    /// @param[in] estride The stride for each entity in active entities.
    /// @param[in] fetch_cells Function that fetches the cell index for an
    /// entity in active_entities (signature:
    /// `std::function<std::int32_t(E::value_type)>`)
    /// @param[in] offset The offset for c
    template <typename T, typename Functor>
    void pack_coefficient_entity(const std::span<T> &c, int cstride,
                                 const dolfinx::fem::Function<T> &u,
                                 const std::span<const std::uint32_t> &cell_info,
                                 const std::span<const std::int32_t> &cells,
                                 std::int32_t num_cell_facets, Functor fetch_cells,
                                 std::int32_t offset, const int codim)
    {
        // Read data from coefficient "u"
        const std::span<const T> &v = u.x()->array();
        const dolfinx::fem::DofMap &dofmap = *u.function_space()->dofmap();
        std::shared_ptr<const dolfinx::fem::FiniteElement> element = u.function_space()->element();
        int space_dim = element->space_dimension();
        const auto transformation = element->get_dof_transformation_function<T>(false, true);

        const int bs = dofmap.bs();
        switch (bs)
        {
            //   case 1:
            //     for (std::size_t e = 0; e < entities.size(); e += estride)
            //     {
            //       auto entity = entities.subspan(e, estride);
            //       std::int32_t cell = fetch_cells(entity);
            //       assert(cell >= 0);
            //       auto cell_coeff = c.subspan(e / estride * cstride + offset, space_dim);
            //       pack<T, 1>(cell_coeff, cell, bs, v, cell_info, dofmap, transformation);
            //     }
            //     break;
            //   case 2:
            //     for (std::size_t e = 0; e < entities.size(); e += estride)
            //     {
            //       auto entity = entities.subspan(e, estride);
            //       std::int32_t cell = fetch_cells(entity);
            //       assert(cell >= 0);
            //       auto cell_coeff = c.subspan(e / estride * cstride + offset, space_dim);
            //       pack<T, 2>(cell_coeff, cell, bs, v, cell_info, dofmap, transformation);
            //     }
            //     break;
            //   case 3:
            //     for (std::size_t e = 0; e < entities.size(); e += estride)
            //     {
            //       auto entity = entities.subspan(e, estride);
            //       std::int32_t cell = fetch_cells(entity);
            //       assert(cell >= 0);
            //       auto cell_coeff = c.subspan(e / estride * cstride + offset, space_dim);
            //       pack<T, 3>(cell_coeff, cell, bs, v, cell_info, dofmap, transformation);
            //     }
            //     break;
        default:
            for (std::int32_t cell = 0; cell < cells.size(); ++cell)
            {
                if (codim == 0)
                {
                }
                else if (codim == 1)
                {
                    for (std::int32_t local_facet = 0; local_facet < num_cell_facets; ++local_facet)
                    {
                        const std::array cell_local_facet = {cell, local_facet};
                        const std::int32_t facet = fetch_cells(cell_local_facet);
                        assert(facet >= 0);

                        // FIXME This may be incorrect for bs > 1 and more than one coefficient
                        auto cell_coeff_f = c.subspan(
                            cell * cstride + local_facet * space_dim + offset,
                            space_dim);
                        dolfinx::fem::impl::pack<T, -1>(cell_coeff_f, facet, bs, v, cell_info, dofmap, transformation);
                    }
                }
                else
                {
                    throw std::runtime_error(
                        "Can't pack coefficients for codimension " + std::to_string(codim) + ".");
                }
            }
            break;
        }
    }

    /// @brief Pack coefficients of a Form for a given integral type and
    /// domain id
    /// @param[in] form The Form
    /// @param[in] integral_type Type of integral
    /// @param[in] id The id of the integration domain
    /// @param[in] c The coefficient array
    /// @param[in] cstride The coefficient stride
    template <typename T>
    void pack_coefficients(const dolfinx::fem::Form<T> &form,
                           dolfinx::fem::IntegralType integral_type, int id,
                           const std::span<T> &c, int cstride)
    {
        if (integral_type != dolfinx::fem::IntegralType::cell)
            throw std::runtime_error(
                "Could not pack coefficient. Integral type not supported.");

        // Get form coefficient offsets and dofmaps
        const std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>>
            &coefficients = form.coefficients();
        const std::vector<int> offsets = form.coefficient_offsets();

        std::shared_ptr<const mesh::Mesh> mesh = form.mesh();
        assert(mesh);
        const int num_cell_facets = mesh::cell_num_entities(
            mesh->topology().cell_type(), mesh->topology().dim() - 1);

        if (!coefficients.empty())
        {
            const std::vector<std::int32_t> &cells = form.cell_domains(id);
            // Iterate over coefficients
            for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
            {
                const int tdim = form.mesh()->topology().dim();
                const int codim = tdim - coefficients[coeff]->function_space()->mesh()->topology().dim();

                auto fetch_cell = form.function_space_to_entity_map(
                    *coefficients[coeff]->function_space());
                // Get cell info for coefficient (with respect to coefficient mesh)
                // NOTE For cell coeffs, need more complicated way of dealing with offset
                std::span<const std::uint32_t> cell_info =
                    dolfinx::fem::impl::get_cell_orientation_info(*coefficients[coeff]);
                dolfinx_hdg::fem::pack_coefficient_entity(c, cstride, *coefficients[coeff],
                                                          cell_info, cells, num_cell_facets, fetch_cell,
                                                          offsets[coeff], codim);
            }
        }
    }

    template <typename T>
    std::vector<int> coefficient_offsets(const dolfinx::fem::Form<T> &form)
    {
        std::vector<int> offsets = {0};
        const std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>>
            &coefficients = form.coefficients();
        std::shared_ptr<const mesh::Mesh> mesh = form.mesh();
        assert(mesh);
        const int tdim = mesh->topology().dim();
        const int num_cell_facets = mesh::cell_num_entities(
            mesh->topology().cell_type(), mesh->topology().dim() - 1);

        for (const auto &coefficient : coefficients)
        {
            if (!coefficient)
                throw std::runtime_error("Not all form coefficients have been set.");
            const int codim = tdim - coefficient->function_space()->mesh()->topology().dim();
            const int ele_space_dim =
                coefficient->function_space()->element()->space_dimension();

            if (codim == 0)
            {
                offsets.push_back(offsets.back() + ele_space_dim);
            }
            else if (codim == 1)
            {
                offsets.push_back(offsets.back() + ele_space_dim * num_cell_facets);
            }
            else
            {
                throw std::runtime_error(
                    "Can't pack coefficients for codimension " + std::to_string(codim) + ".");
            }
        }

        return offsets;
    }

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
        if (integral_type != dolfinx::fem::IntegralType::cell)
            throw std::runtime_error(
                "Could not pack coefficient. Integral type not supported.");

        // Get form coefficient offsets and dofmaps
        const std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>>
            &coefficients = form.coefficients();
        const std::vector<int> offsets = coefficient_offsets(form);

        int num_cells = form.cell_domains(id).size();

        int cstride = 0;
        if (!coefficients.empty())
            cstride = offsets.back();

        return {std::vector<T>(num_cells * cstride), cstride};
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

            dolfinx_hdg::fem::pack_coefficients<T>(form, key.first, key.second, val.first, val.second);
        }
    }
}
