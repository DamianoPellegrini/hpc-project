#pragma once

#include "mst/boruvka/contracts.hpp"

namespace mst::boruvka {

/// Static contract shared by all execution backends.
template <class backend_t>
concept backend = boruvka_round_engine<backend_t>;

} // namespace mst::boruvka
