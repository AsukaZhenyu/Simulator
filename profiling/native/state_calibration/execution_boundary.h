#pragma once
#include <stdexcept>
#include <string>

enum class StateExecutionBoundary {
#define STATE_BOUNDARY(name) name,
#include "execution_boundaries.def"
#undef STATE_BOUNDARY
};
inline const char *state_boundary_name(StateExecutionBoundary value) {
    switch(value) {
#define STATE_BOUNDARY(name) case StateExecutionBoundary::name: return #name;
#include "execution_boundaries.def"
#undef STATE_BOUNDARY
    }
    throw std::runtime_error("invalid execution boundary enum");
}
inline StateExecutionBoundary state_parse_boundary(const std::string &value) {
    if(value.empty())return StateExecutionBoundary::legacy_double_sync;
#define STATE_BOUNDARY(name) if(value==#name)return StateExecutionBoundary::name;
#include "execution_boundaries.def"
#undef STATE_BOUNDARY
    throw std::runtime_error("unknown STATE_PROBE_EXECUTION_BOUNDARY; no fallback");
}
