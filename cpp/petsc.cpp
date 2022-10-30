#include "petsc.h"
#include <iostream>
#include <dolfinx/la/SparsityPattern.h>
#include "utils.h"
#include <dolfinx/la/petsc.h>

Mat dolfinx_hdg::fem::create_matrix(const dolfinx::fem::Form<PetscScalar> &a,
                                    const std::string &type)
{
    //   Build sparsity pattern
    dolfinx::la::SparsityPattern pattern =
        dolfinx_hdg::fem::create_sparsity_pattern(a);

    // Finalise communication
    pattern.assemble();

    return dolfinx::la::petsc::create_matrix(a.mesh()->comm(), pattern, type);
}

// Mat dolfinx_hdg::fem::petsc::create_matrix_block(
void dolfinx_hdg::fem::create_matrix_block(
    const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar> *>> &a,
    const std::string &type)
{
    std::cout << "dolfinx_hdg::fem::create_matrix_block\n";
    // Extract and check row/column ranges
    std::array<std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace>>, 2> V =
        dolfinx::fem::common_function_spaces(extract_function_spaces(a));
    std::array<std::vector<int>, 2> bs_dofs;
    for (std::size_t i = 0; i < 2; ++i)
    {
        for (auto &_V : V[i])
            bs_dofs[i].push_back(_V->dofmap()->bs());
    }

      auto comm = V[0][0]->mesh()->comm();

      // Build sparsity pattern for each block
      std::vector<std::vector<std::unique_ptr<dolfinx::la::SparsityPattern>>> patterns(
          V[0].size());
      for (std::size_t row = 0; row < V[0].size(); ++row)
      {
        for (std::size_t col = 0; col < V[1].size(); ++col)
        {
          const std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> index_maps
              = {{V[0][row]->dofmap()->index_map, V[1][col]->dofmap()->index_map}};
          if (const dolfinx::fem::Form<PetscScalar>* form = a[row][col]; form)
          {
            patterns[row].push_back(std::make_unique<dolfinx::la::SparsityPattern>(
                dolfinx_hdg::fem::create_sparsity_pattern(*form)));
          }
          else
            patterns[row].push_back(nullptr);
        }
      }

    //   // Compute offsets for the fields
    //   std::array<std::vector<std::pair<
    //                  std::reference_wrapper<const common::IndexMap>, int>>,
    //              2>
    //       maps;
    //   for (std::size_t d = 0; d < 2; ++d)
    //   {
    //     for (auto space : V[d])
    //     {
    //       maps[d].emplace_back(*space->dofmap()->index_map,
    //                            space->dofmap()->index_map_bs());
    //     }
    //   }

    //   // Create merged sparsity pattern
    //   std::vector<std::vector<const la::SparsityPattern*>> p(V[0].size());
    //   for (std::size_t row = 0; row < V[0].size(); ++row)
    //     for (std::size_t col = 0; col < V[1].size(); ++col)
    //       p[row].push_back(patterns[row][col].get());
    //   la::SparsityPattern pattern(comm, p, maps, bs_dofs);
    //   pattern.assemble();

    //   // FIXME: Add option to pass customised local-to-global map to PETSc
    //   // Mat constructor

    //   // Initialise matrix
    //   Mat A = la::petsc::create_matrix(comm, pattern, type);

    //   // Create row and column local-to-global maps (field0, field1, field2,
    //   // etc), i.e. ghosts of field0 appear before owned indices of field1
    //   std::array<std::vector<PetscInt>, 2> _maps;
    //   for (int d = 0; d < 2; ++d)
    //   {
    //     // FIXME: Index map concatenation has already been computed inside
    //     // the SparsityPattern constructor, but we also need it here to
    //     // build the PETSc local-to-global map. Compute outside and pass
    //     // into SparsityPattern constructor.

    //     // FIXME: avoid concatenating the same maps twice in case that V[0]
    //     // == V[1].

    //     // Concatenate the block index map in the row and column directions
    //     auto [rank_offset, local_offset, ghosts, _]
    //         = common::stack_index_maps(maps[d]);
    //     for (std::size_t f = 0; f < maps[d].size(); ++f)
    //     {
    //       const common::IndexMap& map = maps[d][f].first.get();
    //       const int bs = maps[d][f].second;
    //       const std::int32_t size_local = bs * map.size_local();
    //       const std::vector global = map.global_indices();
    //       for (std::int32_t i = 0; i < size_local; ++i)
    //         _maps[d].push_back(i + rank_offset + local_offset[f]);
    //       for (std::size_t i = size_local; i < bs * global.size(); ++i)
    //         _maps[d].push_back(ghosts[f][i - size_local]);
    //     }
    //   }

    //   // Create PETSc local-to-global map/index sets and attach to matrix
    //   ISLocalToGlobalMapping petsc_local_to_global0;
    //   ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[0].size(),
    //                                _maps[0].data(), PETSC_COPY_VALUES,
    //                                &petsc_local_to_global0);
    //   if (V[0] == V[1])
    //   {
    //     MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
    //                                petsc_local_to_global0);
    //     ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
    //   }
    //   else
    //   {

    //     ISLocalToGlobalMapping petsc_local_to_global1;
    //     ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[1].size(),
    //                                  _maps[1].data(), PETSC_COPY_VALUES,
    //                                  &petsc_local_to_global1);
    //     MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
    //                                petsc_local_to_global1);
    //     ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
    //     ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);
    //   }

    //   return A;
}
