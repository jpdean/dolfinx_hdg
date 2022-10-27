// #include "utils.h"
// #include <array>
// #include <dolfinx/common/IndexMap.h>
// #include <iostream>

// dolfinx::la::SparsityPattern dolfinx_hdg::fem::create_sparsity_pattern(
//     const dolfinx::mesh::Topology &topology,
//     const std::array<const std::reference_wrapper<const dolfinx::fem::DofMap>, 2> &
//         dofmaps)
// {
//     // Get common::IndexMaps for each dimension
//     const std::array index_maps{dofmaps[0].get().index_map,
//                                 dofmaps[1].get().index_map};
//     const std::array bs = {dofmaps[0].get().index_map_bs(),
//                            dofmaps[1].get().index_map_bs()};

//     // Create and build sparsity pattern
//     assert(dofmaps[0].get().index_map);
//     dolfinx::la::SparsityPattern sp(dofmaps[0].get().index_map->comm(), index_maps,
//                                     bs);

//     const int tdim = topology.dim();
//     auto c_to_f = topology.connectivity(tdim, tdim - 1);
//     assert(c_to_f);
//     for (int c = 0; c < c_to_f->num_nodes(); ++c)
//     {
//         for (auto f_1 : c_to_f->links(c))
//         {
//             for (auto f_2 : c_to_f->links(c))
//             {
//                 sp.insert(dofmaps[0].get().cell_dofs(f_1),
//                           dofmaps[1].get().cell_dofs(f_2));
//             }
//         }
//     }

//     return sp;
// }
