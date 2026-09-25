// ggml-export-workload: turn a small GGML compute graph into an M0 workload
// document (the mapping/examples/*.workload.json schema).
//
// The graph is extracted from the *logical* graph -- before any backend is
// created, before any buffer is allocated, and before a scheduler could split
// the graph across devices. That ordering is deliberate: DESIGN.md 3 asks for
// the logical graph, and a backend would add copies and splits that are not
// part of the workload being modelled.
//
// All string literals in this file are ASCII on purpose. MSVC decodes a source
// file with no BOM using the system code page, so a non-ASCII literal would
// depend on the machine's locale; the artifacts this tool writes must not.
//
// no_alloc = true is used throughout: nothing here computes, and a null `data`
// pointer is not evidence about storage aliasing (see check_layout).

#include "graphs.h"

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace ggml_export {
namespace {

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

// Every refusal names the tensor it is about. An exporter that silently drops,
// renames or reshapes a tensor produces an artifact that still loads, so
// refusing loudly is the only safe behaviour.
struct Failure {
    std::string message;
};

[[noreturn]] void fail(const std::string & message) {
    throw Failure{message};
}

std::string describe(const ggml_tensor * t) {
    if (t == nullptr) {
        return "<null>";
    }
    const char * name = ggml_get_name(t);
    const std::string label = (name != nullptr && name[0] != '\0') ? name : "<unnamed>";
    return "'" + label + "' (op=" + ggml_op_name(t->op) + ")";
}

// ---------------------------------------------------------------------------
// Raw -> normalised semantics
// ---------------------------------------------------------------------------

// M0 models three operations (spec.py: _SEMANTIC_OPS). Anything else is an
// unsupported semantic and must be refused rather than passed through: a
// guessed name would silently turn into a different workload.
struct Semantics {
    const char * semantic_op;
    bool is_unary;  // true only for GGML_OP_UNARY, which carries a unary tag
    int op_params_len;
};

Semantics semantics_of(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_MUL_MAT:
            return {"MUL_MAT", false, 0};
        case GGML_OP_ADD:
            return {"ADD", false, 0};
        case GGML_OP_UNARY: {
            // The trap: ggml_relu() is not an op of its own. It is GGML_OP_UNARY
            // carrying a unary tag, so a reader that looks only at `op` cannot
            // tell relu from any other unary. Both forms are recorded: the raw
            // tag and the normalised name.
            const enum ggml_unary_op unary = ggml_get_unary_op(t);
            if (unary == GGML_UNARY_OP_RELU) {
                // op_params[0] carries the tag. A non-zero value there is legal
                // and is not a reason to reject the tensor.
                return {"RELU", true, 1};
            }
            fail("unsupported unary op on " + describe(t) + ": " + ggml_unary_op_name(unary));
        }
        default:
            break;
    }
    fail("unsupported op on " + describe(t) + ": " + ggml_op_name(t->op));
}

int input_count(const ggml_tensor * t) {
    int count = 0;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i] != nullptr) {
            ++count;
        }
    }
    return count;
}

// ggml's own *_name() functions return the enumerator without its prefix:
// ggml_op_name(GGML_OP_MUL_MAT) is "MUL_MAT" and ggml_unary_op_name returns
// "RELU". The M0 schema spells the C enumerator in full, so prefixing is exact
// -- and unlike a hand-written table it follows the library if an enum is
// renamed.
std::string op_name(enum ggml_op op) {
    return std::string("GGML_OP_") + ggml_op_name(op);
}

std::string unary_op_name(enum ggml_unary_op op) {
    return std::string("GGML_UNARY_OP_") + ggml_unary_op_name(op);
}

// dtype is the exception: ggml_type_name() returns the lowercase short name
// ("f32") while the schema wants "GGML_TYPE_F32", so the prefix trick does not
// apply. check_layout has already refused every other type, so an F32-only
// answer is exhaustive by construction.
const char * dtype_name(enum ggml_type type) {
    if (type != GGML_TYPE_F32) {
        fail("unsupported dtype in emission: " + std::string(ggml_type_name(type)));
    }
    return "GGML_TYPE_F32";
}

// ---------------------------------------------------------------------------
// Layout checks
// ---------------------------------------------------------------------------

// One storage tensor = one independent allocation. `view_src` is how ggml
// expresses sharing and in-place results, so a non-null view_src means the
// tensor does not own its bytes and M0's one-allocation-per-tensor model does
// not hold. This cannot be checked by comparing `data` pointers: with
// no_alloc = true every data pointer is null, and that says nothing about
// aliasing.
int64_t check_layout(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32) {
        fail("unsupported dtype on " + describe(t) + ": " + ggml_type_name(t->type) +
             " (M0 models F32 only)");
    }
    if (t->view_src != nullptr) {
        fail("unsupported view on " + describe(t) +
             ": view_src is set, so this tensor does not own its storage");
    }

    // Contiguity, read from the real nb (ggml order: ne[0] is innermost, never
    // numpy order) and cross-checked against ggml_nbytes.
    const size_t type_size = ggml_type_size(t->type);
    if (static_cast<size_t>(t->nb[0]) != type_size) {
        fail("non-contiguous " + describe(t) + ": nb[0] is " + std::to_string(t->nb[0]) +
             ", expected the F32 type size " + std::to_string(type_size));
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        const int64_t expected = t->nb[i - 1] * t->ne[i - 1];
        if (t->nb[i] != expected) {
            fail("non-contiguous " + describe(t) + ": nb[" + std::to_string(i) + "] is " +
                 std::to_string(t->nb[i]) + ", expected nb[" + std::to_string(i - 1) +
                 "]*ne[" + std::to_string(i - 1) + "] = " + std::to_string(expected));
        }
    }

    const int64_t storage_bytes = static_cast<int64_t>(ggml_nbytes(t));
    const int64_t implied = t->nb[GGML_MAX_DIMS - 1] * t->ne[GGML_MAX_DIMS - 1];
    if (storage_bytes != implied) {
        fail("ggml_nbytes disagrees with nb*ne for " + describe(t) + ": " +
             std::to_string(storage_bytes) + " vs " + std::to_string(implied));
    }
    if (storage_bytes <= 0) {
        fail("zero-sized tensor " + describe(t));
    }
    return storage_bytes;
}

// ---------------------------------------------------------------------------
// Planning: the ancestor closure of the requested outputs
// ---------------------------------------------------------------------------

struct Declaration {
    const Graph * graph{nullptr};
    std::map<const ggml_tensor *, const Leaf *> leaf_of;
    std::map<const ggml_tensor *, const Node *> node_of;
};

void collect(const ggml_tensor * t, std::unordered_set<const ggml_tensor *> & reachable) {
    if (!reachable.insert(t).second) {
        return;
    }
    // Dependencies come from src[] and nothing else. A cgraph's node array is a
    // traversal order, not a set of dependency edges; treating it as one would
    // invent edges the graph does not have.
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i] != nullptr) {
            collect(t->src[i], reachable);
        }
    }
}

void plan_graph(const Graph & g, Declaration & decl) {
    const std::string where = "graph '" + std::string(g.id) + "': ";

    decl.graph = &g;
    for (const Leaf & leaf : g.leaves) {
        if (leaf.tensor == nullptr) {
            fail(where + "leaf '" + leaf.id + "' was not created -- out of context memory?");
        }
        if (leaf.tensor->op != GGML_OP_NONE) {
            fail(where + "'" + leaf.id + "' is declared as a leaf but its op is " +
                 ggml_op_name(leaf.tensor->op) +
                 " (a leaf must be GGML_OP_NONE, so it produces no compute action)");
        }
        if (!decl.leaf_of.emplace(leaf.tensor, &leaf).second) {
            fail(where + "duplicate leaf '" + leaf.id + "'");
        }
    }

    for (const Node & node : g.nodes) {
        if (node.tensor == nullptr) {
            fail(where + "node '" + node.op_id + "' was not created -- out of context memory?");
        }
        if (node.tensor->op == GGML_OP_NONE) {
            fail(where + "node '" + node.op_id + "' is declared as computed but its op is NONE");
        }
        if (decl.leaf_of.count(node.tensor) != 0) {
            fail(where + "'" + node.op_id + "' is declared as both a leaf and a node");
        }
        if (!decl.node_of.emplace(node.tensor, &node).second) {
            fail(where + "duplicate node '" + node.op_id + "'");
        }
    }

    std::unordered_set<const ggml_tensor *> reachable;
    for (const ggml_tensor * output : g.outputs) {
        if (output == nullptr) {
            fail(where + "a requested output was not created");
        }
        collect(output, reachable);
    }

    for (const ggml_tensor * t : reachable) {
        if (decl.leaf_of.count(t) == 0 && decl.node_of.count(t) == 0) {
            fail(where + describe(t) + " is an ancestor of a requested output but was never declared");
        }
    }
    // Export only the ancestor closure: a declaration the outputs cannot reach
    // would put a tensor in the artifact that the workload never uses.
    for (const Leaf & leaf : g.leaves) {
        if (reachable.count(leaf.tensor) == 0) {
            fail(where + "leaf '" + leaf.id + "' is not an ancestor of the requested outputs");
        }
    }
    for (const Node & node : g.nodes) {
        if (reachable.count(node.tensor) == 0) {
            fail(where + "node '" + node.op_id + "' is not an ancestor of the requested outputs");
        }
        const int arity = input_count(node.tensor);
        const int expected = std::strcmp(semantics_of(node.tensor).semantic_op, "RELU") == 0 ? 1 : 2;
        if (arity != expected) {
            fail(where + "node '" + node.op_id + "' has " + std::to_string(arity) +
                 " inputs, expected " + std::to_string(expected));
        }
    }

    // The declarations must already be ordered so that every input precedes its
    // consumer. Emitting in declaration order then yields a document whose
    // tensor and operation lists read topologically with no re-sorting.
    std::unordered_set<const ggml_tensor *> emitted;
    for (const Leaf & leaf : g.leaves) {
        emitted.insert(leaf.tensor);
    }
    for (const Node & node : g.nodes) {
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            const ggml_tensor * src = node.tensor->src[i];
            if (src != nullptr && emitted.count(src) == 0) {
                fail(where + "node '" + node.op_id + "' reads " + describe(src) +
                     " which is declared after it; the graph table is not in topological order");
            }
        }
        emitted.insert(node.tensor);
    }
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

std::string quote(const std::string & in) {
    std::string out = "\"";
    for (const char raw : in) {
        const unsigned char c = static_cast<unsigned char>(raw);
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    out += buffer;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    out += "\"";
    return out;
}

// Templated because ggml does not use one integer type for both arrays: `ne` is
// int64_t[] and `nb` is size_t[].
template <typename T>
std::string csv_ints(const T * values, int count) {
    std::string out = "[";
    for (int i = 0; i < count; ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += std::to_string(values[i]);
    }
    out += "]";
    return out;
}

// Indentation and key order follow json.dumps(..., indent=2) so an export and a
// hand-written fixture can be compared with a plain diff.
std::string tensor_entry(const std::string & id, const std::string & role, const ggml_tensor * t,
                         int64_t storage_bytes, const char * initial) {
    std::string out = "    {\n";
    out += "      \"id\": " + quote(id) + ",\n";
    out += "      \"name\": " + quote(id) + ",\n";
    out += "      \"role\": " + quote(role) + ",\n";
    out += "      \"dtype\": " + quote(dtype_name(t->type)) + ",\n";
    out += "      \"ne\": " + csv_ints(t->ne, GGML_MAX_DIMS) + ",\n";
    out += "      \"nb\": " + csv_ints(t->nb, GGML_MAX_DIMS) + ",\n";
    if (initial != nullptr) {
        out += "      \"storage_bytes\": " + std::to_string(storage_bytes) + ",\n";
        out += "      \"initial_locations\": [" + quote(initial) + "]\n";
    } else {
        // Computed tensors have no initial location: they come into existence
        // when their operation runs. The key is omitted rather than nulled.
        out += "      \"storage_bytes\": " + std::to_string(storage_bytes) + "\n";
    }
    out += "    }";
    return out;
}

std::string operation_entry(const Node & node, const std::string & output_id) {
    const Semantics s = semantics_of(node.tensor);
    std::string out = "    {\n";
    out += "      \"id\": " + quote(node.op_id) + ",\n";
    out += "      \"semantic_op\": " + quote(s.semantic_op) + ",\n";
    out += "      \"ggml_op\": " + quote(op_name(node.tensor->op)) + ",\n";
    if (s.is_unary) {
        out += "      \"unary_op\": " + quote(unary_op_name(ggml_get_unary_op(node.tensor))) + ",\n";
    } else {
        out += "      \"unary_op\": null,\n";
    }
    // Only the op_params entries this op actually uses. The remaining slots are
    // unused padding, and dumping all GGML_MAX_OP_PARAMS of them would make the
    // artifact depend on a buffer's uninitialised tail.
    out += "      \"op_params\": [";
    for (int i = 0; i < s.op_params_len; ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += std::to_string(node.tensor->op_params[i]);
    }
    out += "],\n";
    out += "      \"inputs\": [";
    bool first = true;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = node.tensor->src[i];
        if (src == nullptr) {
            continue;
        }
        if (!first) {
            out += ", ";
        }
        first = false;
        out += quote(ggml_get_name(src));
    }
    out += "],\n";
    out += "      \"output\": " + quote(output_id) + "\n";
    out += "    }";
    return out;
}

std::string join_entries(const std::vector<std::string> & entries) {
    std::string out;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i != 0) {
            out += ",\n";
        }
        out += entries[i];
    }
    return out;
}

std::string tensor_id(const ggml_tensor * t) {
    const char * name = ggml_get_name(t);
    return (name != nullptr && name[0] != '\0') ? name : "";
}

std::string emit_document(const Graph & g,
                          const std::map<const ggml_tensor *, int64_t> & sizes,
                          const std::string & version, const std::string & commit) {
    const std::string comment =
        "Real GGML export (origin.kind = ggml): ggml " + version + ", commit " + commit +
        ". F32, contiguous, no views, no in-place sharing; one storage tensor = one "
        "independent allocation. Extracted from the logical graph via "
        "ggml_build_forward_expand before any backend allocation or split. Build method and "
        "source tree are recorded in mapping/ggml/README.md.";

    // Leaves first, then computed tensors, each group in declaration order. The
    // declarations are already topological (plan_graph checked), so this is
    // also the order the model needs: a reader can walk it front to back.
    std::vector<std::string> tensors;
    for (const Leaf & leaf : g.leaves) {
        tensors.push_back(tensor_entry(leaf.id, leaf.role, leaf.tensor, sizes.at(leaf.tensor),
                                       leaf.initial));
    }
    for (const Node & node : g.nodes) {
        bool is_output = false;
        for (const ggml_tensor * o : g.outputs) {
            is_output = is_output || (o == node.tensor);
        }
        tensors.push_back(tensor_entry(tensor_id(node.tensor), is_output ? "output" : "intermediate",
                                       node.tensor, sizes.at(node.tensor), nullptr));
    }

    std::vector<std::string> operations;
    for (const Node & node : g.nodes) {
        operations.push_back(operation_entry(node, tensor_id(node.tensor)));
    }

    std::string outputs = "[";
    for (size_t i = 0; i < g.outputs.size(); ++i) {
        if (i != 0) {
            outputs += ", ";
        }
        outputs += quote(tensor_id(g.outputs[i]));
    }
    outputs += "]";

    std::string out;
    out += "{\n";
    out += "  \"schema_version\": \"0.1\",\n";
    out += "  \"id\": " + quote(g.id) + ",\n";
    out += "  \"origin\": {\n";
    out += "    \"kind\": \"ggml\",\n";
    out += "    \"ggml_version\": " + quote(version) + ",\n";
    out += "    \"sample\": " + quote(g.sample) + "\n";
    out += "  },\n";
    out += "  \"comment\": " + quote(comment) + ",\n";
    out += "  \"tensors\": [\n" + join_entries(tensors) + "\n  ],\n";
    out += "  \"operations\": [\n" + join_entries(operations) + "\n  ],\n";
    out += "  \"outputs\": " + outputs + "\n";
    out += "}\n";
    return out;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const char * kUsage =
    "usage: ggml-export-workload --graph <chain|residual|fork|matvec|all>\n"
    "                            (--out FILE | --out-dir DIR)\n"
    "The negative fixtures reject-f16 / reject-op / reject-unary / reject-inplace\n"
    "are accepted by name but are never part of 'all': each one violates exactly\n"
    "one layout rule and must exit non-zero naming the offending tensor.\n";

// A name the CLI accepts, whether or not it is exportable.
bool is_known_graph(const std::string & name) {
    for (const char * known : graph_names()) {
        if (name == known) {
            return true;
        }
    }
    for (const char * known : rejection_graph_names()) {
        if (name == known) {
            return true;
        }
    }
    return false;
}

struct Options {
    std::vector<std::string> graphs;
    std::string out_file;
    std::string out_dir;
};

Options parse_args(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        const bool has_value = (i + 1) < argc;
        if (flag == "--graph" && has_value) {
            options.graphs.emplace_back(argv[++i]);
        } else if (flag == "--out" && has_value) {
            options.out_file = argv[++i];
        } else if (flag == "--out-dir" && has_value) {
            options.out_dir = argv[++i];
        } else if (flag == "--help" || flag == "-h") {
            std::fputs(kUsage, stdout);
            std::exit(0);
        } else {
            fail("unrecognised argument '" + flag + "'\n" + kUsage);
        }
    }
    if (options.graphs.empty()) {
        fail(std::string("--graph is required\n") + kUsage);
    }
    if (options.out_file.empty() && options.out_dir.empty()) {
        fail(std::string("one of --out or --out-dir is required\n") + kUsage);
    }
    if (!options.out_file.empty() && !options.out_dir.empty()) {
        fail(std::string("--out and --out-dir are mutually exclusive\n") + kUsage);
    }
    return options;
}

}  // namespace
}  // namespace ggml_export

int main(int argc, char ** argv) {
    using namespace ggml_export;

    try {
        const Options options = parse_args(argc, argv);

        const std::string version = ggml_version();
        const std::string commit = ggml_commit();
        std::fprintf(stderr, "ggml-export-workload: ggml %s commit %s\n", version.c_str(),
                     commit.c_str());

        // Expand "all" and reject anything else before touching the filesystem.
        std::vector<std::string> wanted;
        for (const std::string & name : options.graphs) {
            if (name == "all") {
                for (const char * known : graph_names()) {
                    wanted.emplace_back(known);
                }
                continue;
            }
            bool is_known = is_known_graph(name);
            if (!is_known) {
                fail("unknown --graph '" + name + "'\n" + kUsage);
            }
            wanted.emplace_back(name);
        }
        if (wanted.size() != 1 && !options.out_file.empty()) {
            fail("--out needs exactly one --graph");
        }

        // Enough for every tensor of one graph plus its cgraph. no_alloc means
        // no tensor data is ever allocated here, so this pool stays tiny.
        const size_t mem_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
        std::fprintf(stderr, "  context: tensor_overhead=%llu graph_overhead=%llu mem_size=%llu\n",
                     static_cast<unsigned long long>(ggml_tensor_overhead()),
                     static_cast<unsigned long long>(ggml_graph_overhead()),
                     static_cast<unsigned long long>(mem_size));

        for (const std::string & name : wanted) {
            // One context per graph: an export does not depend on any other
            // export, and a shared pool would make each artifact depend on the
            // order the graphs happened to be written in.
            ggml_init_params params = {
                /*.mem_size   =*/ mem_size,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            if (ctx == nullptr) {
                fail("ggml_init failed");
            }

            const Graph g = build_graph(ctx, name.c_str());
            if (g.id == nullptr) {
                fail("unknown graph '" + name + "'");
            }

            // Build the graph through the real API first. This is what proves
            // the graph is well formed, and it is the entry point DESIGN.md 3
            // names; the export itself then walks src[] rather than the node
            // array, because struct ggml_cgraph is only an opaque forward
            // declaration in ggml.h.
            ggml_cgraph * gf = ggml_new_graph(ctx);
            for (ggml_tensor * output : g.outputs) {
                ggml_build_forward_expand(gf, output);
            }
            const int reported = ggml_graph_n_nodes(gf);

            Declaration decl;
            plan_graph(g, decl);

            // Diagnostic, and a guard in the direction that matters: ggml seeing
            // fewer computed nodes than the exporter declares would mean the
            // exported graph is not the graph that was built.
            std::fprintf(stderr, "  %-9s ggml reports %d graph nodes, exporter declares %zu\n",
                         g.id, reported, g.nodes.size());
            if (reported < static_cast<int>(g.nodes.size())) {
                fail("graph '" + std::string(g.id) + "': ggml reports " + std::to_string(reported) +
                     " nodes but the exporter declares " + std::to_string(g.nodes.size()));
            }

            std::map<const ggml_tensor *, int64_t> sizes;
            for (const Leaf & leaf : g.leaves) {
                sizes[leaf.tensor] = check_layout(leaf.tensor);
            }
            for (const Node & node : g.nodes) {
                sizes[node.tensor] = check_layout(node.tensor);
            }

            const std::string document = emit_document(g, sizes, version, commit);

            std::filesystem::path target;
            if (!options.out_file.empty()) {
                target = options.out_file;
            } else {
                target = std::filesystem::path(options.out_dir) /
                         (std::string(g.id) + ".workload.json");
                std::filesystem::create_directories(options.out_dir);
            }

            // Binary mode: the document uses '\n' and must not be rewritten to
            // CRLF by the C runtime, so a re-run compares byte for byte.
            std::ofstream stream(target, std::ios::binary | std::ios::trunc);
            if (!stream) {
                fail("cannot write " + target.string());
            }
            stream << document;
            stream.close();
            if (!stream) {
                fail("failed while writing " + target.string());
            }

            std::fprintf(stderr, "  %-9s wrote %s\n", g.id, target.string().c_str());

            // Freeing the context invalidates g, decl and sizes, so the file is
            // written above and nothing below may read them.
            ggml_free(ctx);
        }
        return 0;
    } catch (const Failure & failure) {
        std::fprintf(stderr, "error: %s\n", failure.message.c_str());
        return 1;
    }
}
