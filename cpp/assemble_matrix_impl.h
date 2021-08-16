#pragma once

// FIXME See which of these aren't needed
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <iterator>
#include <vector>
#include <xtl/xspan.hpp>
#include <dolfinx/common/array2d.h>
#include <iostream>

namespace dolfinx_hdg::fem::impl
{
    template <typename T>
    void assemble_matrix(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_set,
        const dolfinx::fem::Form<T> &a, const xtl::span<const T> &constants,
        const dolfinx::array2d<T> &coeffs, const std::vector<bool> &bc0,
        const std::vector<bool> &bc1);

    template <typename T>
    void assemble_matrix(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_set,
        const dolfinx::fem::Form<T> &a, const xtl::span<const T> &constants,
        const dolfinx::array2d<T> &coeffs, const std::vector<bool> &bc0,
        const std::vector<bool> &bc1)
    {
        std::cout << "Hello from dolfinx_hdg::fem::impl::assemble_matrix\n";
    }
}
