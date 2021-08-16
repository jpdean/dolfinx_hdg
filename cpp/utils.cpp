#include "utils.h"
#include <array>
#include <dolfinx/common/IndexMap.h>

dolfinx::la::SparsityPattern dolfinx_hdg::fem::create_sparsity_pattern(
    const dolfinx::mesh::Topology &topology,
    const std::array<const std::reference_wrapper<const dolfinx::fem::DofMap>, 2> &
        dofmaps)
{
    // Get common::IndexMaps for each dimension
    const std::array index_maps{dofmaps[0].get().index_map,
                                dofmaps[1].get().index_map};
    const std::array bs = {dofmaps[0].get().index_map_bs(),
                           dofmaps[1].get().index_map_bs()};

    // Create and build sparsity pattern
    assert(dofmaps[0].get().index_map);
    dolfinx::la::SparsityPattern sp(
        dofmaps[0].get().index_map->comm(
            dolfinx::common::IndexMap::Direction::forward),
        index_maps, bs);

    const int D = topology.dim() - 1;
    auto cells = topology.connectivity(D, 0);
    assert(cells);
    for (int c = 0; c < cells->num_nodes(); ++c)
    {
        sp.insert(dofmaps[0].get().cell_dofs(c),
                  dofmaps[1].get().cell_dofs(c));
    }

    // TODO Assemble here or return unassembled?
    sp.assemble();
    return sp;
}
