#include "assembler_helpers.h"
#include <xtensor/xview.hpp>

void dolfinx_hdg::fem::impl_helpers::get_coordinate_dofs(
    std::vector<double> &coordinate_dofs,
    const std::int32_t cell,
    const dolfinx::graph::AdjacencyList<int32_t> &x_dofmap,
    const xt::xtensor<double, 2> &x_g)
{
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
        std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                    std::next(coordinate_dofs.begin(), 3 * i));
    }
}

void dolfinx_hdg::fem::impl_helpers::get_cell_facet_perms(
    std::vector<std::uint8_t> &cell_facet_perms,
    const std::int32_t cell,
    const int num_cell_facets,
    const std::function<std::uint8_t(std::size_t)> &get_perm)
{
    for (int local_f = 0; local_f < num_cell_facets; ++local_f)
    {
        cell_facet_perms[local_f] =
            get_perm(cell * num_cell_facets + local_f);
    }
}
