// // TODO Rename

// #include "static_condensation_helpers.h"

// // TODO Put in .cpp
// std::vector<double> dolfinx_hdg::sc::get_cell_coord_dofs(
//     std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
//     const int cell)
// {
//     const dolfinx::graph::AdjacencyList<std::int32_t> &x_dofmap =
//         mesh->geometry().dofmap();
//     const std::size_t num_dofs_g = x_dofmap.num_links(cell);
//     const xt::xtensor<double, 2> &x_g = mesh->geometry().x();
//     std::vector<double> coordinate_dofs(3 * num_dofs_g);
//     auto x_dofs = x_dofmap.links(cell);
//     for (std::size_t i = 0; i < x_dofs.size(); ++i)
//     {
//         std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
//                     std::next(coordinate_dofs.begin(), 3 * i));
//     }
//     return coordinate_dofs;
// }

// std::vector<double> dolfinx_hdg::sc::get_facet_coord_dofs(
//     std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
//     const int f)
// {
//     // FIXME Get facet coord dofs properly and check
//     const int tdim = mesh->topology().dim();
//     auto facet_x_dofs = mesh->topology().connectivity(tdim - 1, tdim - 2)->links(f);
//     const xt::xtensor<double, 2> &x_g = mesh->geometry().x();
//     std::vector<double> fact_coordinate_dofs(3 * mesh->topology().connectivity(tdim - 1, tdim - 2)->num_links(0));
//     for (std::size_t i = 0; i < facet_x_dofs.size(); ++i)
//     {
//         std::copy_n(xt::row(x_g, facet_x_dofs[i]).begin(), 3,
//                     std::next(fact_coordinate_dofs.begin(), 3 * i));
//     }
//     return fact_coordinate_dofs;
// }