#pragma once

#include "mst/boruvka/contracts.hpp"

namespace mst::boruvka {

/// Alias comodo su `boruvka_round_engine`: ogni backend deve rispettare
/// questo contratto a tempo di compilazione.
template <class backend_t>
concept backend = boruvka_round_engine<backend_t>;

} // namespace mst::boruvka
