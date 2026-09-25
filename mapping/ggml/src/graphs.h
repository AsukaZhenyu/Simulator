// Graph descriptions handed to the exporter.
//
// The exporter does not guess anything about a graph: the builder that calls
// ggml_* also declares, for every tensor it created, whether that tensor is a
// leaf and what role it plays. A ggml graph cannot answer that question on its
// own -- nothing in a ggml_tensor says whether a weight lives in DRAM or
// whether an activation is already resident -- so the declaration is explicit
// rather than inferred from shape or from the op that consumes it.
#pragma once

#include "ggml.h"

#include <vector>

namespace ggml_export {

// One leaf of the graph (op == GGML_OP_NONE): a weight or an input. ``role`` and
// ``initial`` are the M0 vocabulary (spec.py: _ROLES / _LOCATIONS).
struct Leaf {
    const char * id;
    const char * role;     // "weight" | "input"
    const char * initial;  // "dram" | "vram"
    ggml_tensor * tensor;
};

// One computed tensor. ggml has no notion of an operation id separate from the
// tensor the operation produces, so the builder supplies one; without it the
// mapping document could not name an operation to schedule.
struct Node {
    const char * op_id;
    ggml_tensor * tensor;
};

struct Graph {
    const char * id;
    const char * sample;
    std::vector<Leaf> leaves;
    std::vector<Node> nodes;  // construction order, which is topological
    std::vector<ggml_tensor *> outputs;
};

// Every graph the exporter can emit, in export order. These are what --graph all
// expands to.
std::vector<const char *> graph_names();

// Graphs that exist only to be refused: each one violates exactly one of the
// layout rules the exporter must enforce. They are reachable by name so that
// tests can observe the refusal firing, but they are deliberately not part of
// graph_names() -- an export run must never emit them. See the builders for
// which rule each one breaks.
std::vector<const char *> rejection_graph_names();

// Builds one named graph into an already-initialised context. Tensor and
// operation ids deliberately match the checked-in fixtures in
// mapping/examples/, so an export and its fixture can be compared field by
// field instead of through a renaming table. Returns a Graph whose id is null
// if the name is not one of graph_names().
Graph build_graph(ggml_context * ctx, const char * name);

}  // namespace ggml_export
