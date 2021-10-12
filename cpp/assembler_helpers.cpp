#include "assembler_helpers.h"
#include <xtensor/xio.hpp> // To cout xtensor arrays
#include <xtensor/xview.hpp>

void dolfinx_hdg::fem::impl_helpers::get_coordinate_dofs(
    std::vector<double>& coordinate_dofs,
    const std::int32_t cell,
    const xtl::span<const int>& cell_facets,
    const dolfinx::graph::AdjacencyList<int32_t>& x_dofmap,
    const xt::xtensor<double, 2>& x_g,
    xt::xtensor<int32_t, 2> ent_to_geom)
{
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
        std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                    std::next(coordinate_dofs.begin(), 3 * i));
    }

    // std::cout << "ent_to_geom = \n";
    // std::cout << ent_to_geom << "\n";

    int offset = 3 * x_dofs.size();
    for (int local_facet = 0; local_facet < cell_facets.size(); ++local_facet)
    {
        // std::cout << "local_facet = " << local_facet << "\n";
        auto facet_x_dofs = xt::row(ent_to_geom, local_facet);
        // std::cout << "facet_x_dofs = " << facet_x_dofs << "\n";

        for (std::size_t i = 0; i < facet_x_dofs.size(); ++i)
        {
            // std::cout << "i = " << i << "\n";
            // std::cout << "facet_x_dofs[i] = " << facet_x_dofs[i] << "\n";
            
            std::copy_n(xt::row(x_g, facet_x_dofs[i]).begin(), 3,
                        std::next(coordinate_dofs.begin(), offset + 3 * i));
        }
        offset += 3 * facet_x_dofs.size();
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
