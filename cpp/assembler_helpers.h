#pragma once

#include <dolfinx/graph/AdjacencyList.h>
#include <xtensor/xtensor.hpp>
#include <vector>

namespace dolfinx_hdg::fem::impl_helpers
{
    void get_coordinate_dofs(
        std::vector<double>& coordinate_dofs,
        const std::int32_t cell,
        const xtl::span<const int>& cell_facets,
        const dolfinx::graph::AdjacencyList<int32_t>& x_dofmap,
        const xt::xtensor<double, 2>& x_g,
        xt::xtensor<int32_t, 2> ent_to_geom);

    void get_cell_facet_perms(
        std::vector<std::uint8_t>& cell_facet_perms,
        const std::int32_t cell,
        const tcb::span<const int32_t> &cell_facets,
        const std::function<std::uint8_t(std::size_t)>& get_perm,
        const std::function<std::uint8_t(std::size_t)> &get_full_cell_perm);
}
