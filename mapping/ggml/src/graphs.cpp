#include "graphs.h"

#include <string>

namespace ggml_export {
namespace {

// Shapes are written the ggml way, not the numpy way: ggml_new_tensor_2d(ctx, T,
// ne0, ne1) gives ne = {ne0, ne1, 1, 1} and, for F32, nb = {4, 4*ne0, ...}.
// MUL_MAT takes [weight, activation] with weight ne = [in_features,
// out_features, 1, 1] (spec.py validates exactly that), so a 4x4 projection is
// ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4).

ggml_tensor * make(ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1) {
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    // Named explicitly so the export is reproducible: the M0 id is the ggml
    // name, never a pointer address.
    ggml_set_name(tensor, name);
    return tensor;
}

// chain: h = W1*x; y = W2*h  (the two-layer dependency chain)
Graph make_chain(ggml_context * ctx) {
    ggml_tensor * W1 = make(ctx, "W1", 4, 4);
    ggml_tensor * W2 = make(ctx, "W2", 4, 4);
    ggml_tensor * x = make(ctx, "x", 4, 1);
    ggml_tensor * h = ggml_mul_mat(ctx, W1, x);
    ggml_set_name(h, "h");
    ggml_tensor * y = ggml_mul_mat(ctx, W2, h);
    ggml_set_name(y, "y");

    Graph g{"chain", "chain: h = W1*x; y = W2*h", {}, {}, {}};
    g.leaves = {
        {"W1", "weight", "dram", W1},
        {"W2", "weight", "dram", W2},
        {"x", "input", "vram", x},
    };
    g.nodes = {{"c1", h}, {"c2", y}};
    g.outputs = {y};
    return g;
}

Graph make_residual(ggml_context * ctx) {
    ggml_tensor * W1 = make(ctx, "W1", 4, 4);
    ggml_tensor * W2 = make(ctx, "W2", 4, 4);
    ggml_tensor * x = make(ctx, "x", 4, 1);
    ggml_tensor * h = ggml_mul_mat(ctx, W1, x);
    ggml_set_name(h, "h");
    // ggml_relu() is GGML_OP_UNARY plus the GGML_UNARY_OP_RELU tag -- not an op
    // of its own. The export keeps both the raw and the normalised form.
    ggml_tensor * a = ggml_relu(ctx, h);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_mul_mat(ctx, W2, a);
    ggml_set_name(b, "b");
    ggml_tensor * y = ggml_add(ctx, b, x);
    ggml_set_name(y, "y");

    Graph g{"residual",
            "residual: h = W1*x; a = relu(h); b = W2*a; y = b + x",
            {},
            {},
            {}};
    g.leaves = {
        {"W1", "weight", "dram", W1},
        {"W2", "weight", "dram", W2},
        {"x", "input", "vram", x},
    };
    g.nodes = {{"c1", h}, {"r", a}, {"c2", b}, {"add", y}};
    g.outputs = {y};
    return g;
}

Graph make_fork(ggml_context * ctx) {
    ggml_tensor * W1 = make(ctx, "W1", 4, 4);
    ggml_tensor * W2 = make(ctx, "W2", 4, 4);
    ggml_tensor * x = make(ctx, "x", 4, 1);
    ggml_tensor * a = ggml_mul_mat(ctx, W1, x);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_mul_mat(ctx, W2, x);
    ggml_set_name(b, "b");
    ggml_tensor * y = ggml_add(ctx, a, b);
    ggml_set_name(y, "y");

    Graph g{"fork", "fork: a = W1*x; b = W2*x; y = a + b", {}, {}, {}};
    g.leaves = {
        {"W1", "weight", "dram", W1},
        {"W2", "weight", "dram", W2},
        {"x", "input", "vram", x},
    };
    g.nodes = {{"c1", a}, {"c2", b}, {"add", y}};
    g.outputs = {y};
    return g;
}

// A rectangular MUL_MAT. A square weight hides an ne/nb convention error: swap
// in_features and out_features and a square matrix still produces the same byte
// counts. Here the weight is 3x4 and the activation is 3x1, so misreading the
// layout as 4x3 would demand a 4-wide activation (16 B, not 12 B) and every
// downstream number would shift.
Graph make_matvec(ggml_context * ctx) {
    ggml_tensor * W = make(ctx, "W", 3, 4);
    ggml_tensor * x = make(ctx, "x", 3, 1);
    ggml_tensor * y = ggml_mul_mat(ctx, W, x);
    ggml_set_name(y, "y");

    Graph g{"matvec", "rectangular MUL_MAT: y = W*x with in=3, out=4", {}, {}, {}};
    g.leaves = {
        {"W", "weight", "dram", W},
        {"x", "input", "vram", x},
    };
    g.nodes = {{"c1", y}};
    g.outputs = {y};
    return g;
}

// ---------------------------------------------------------------------------
// Negative fixtures.
//
// The exporter refuses four things (see check_layout / semantics_of in
// export_workload.cpp). A refusal that no test ever triggers is only a claim
// that the code contains a branch, so each rule gets a graph that breaks it.
// These build successfully as ggml graphs -- the point is that the graph is
// legal but the *export* is not -- and every one of them must die with a
// non-zero exit naming the offending tensor.
// ---------------------------------------------------------------------------

// Breaks: every tensor must be GGML_TYPE_F32. An F16 weight is a perfectly
// ordinary ggml graph (llama.cpp builds them constantly), so nothing upstream
// stops it; the exporter is the only thing that does.
Graph make_reject_f16(ggml_context * ctx) {
    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4, 4);
    ggml_set_name(W, "W");
    ggml_tensor * x = make(ctx, "x", 4, 1);
    ggml_tensor * y = ggml_mul_mat(ctx, W, x);
    ggml_set_name(y, "y");

    Graph g{"reject-f16", "negative fixture: an F16 weight", {}, {}, {}};
    g.leaves = {
        {"W", "weight", "dram", W},
        {"x", "input", "vram", x},
    };
    g.nodes = {{"c1", y}};
    g.outputs = {y};
    return g;
}

// Breaks: the op must be one of the three M0 models. ggml_dup is an ordinary
// op; M0 has no semantic for it, and guessing one would silently mis-model the
// cost.
Graph make_reject_op(ggml_context * ctx) {
    ggml_tensor * W1 = make(ctx, "W1", 4, 4);
    ggml_tensor * x = make(ctx, "x", 4, 1);
    ggml_tensor * h = ggml_mul_mat(ctx, W1, x);
    ggml_set_name(h, "h");
    ggml_tensor * y = ggml_dup(ctx, h);
    ggml_set_name(y, "y");

    Graph g{"reject-op", "negative fixture: GGML_OP_DUP is not modelled", {}, {}, {}};
    g.leaves = {
        {"W1", "weight", "dram", W1},
        {"x", "input", "vram", x},
    };
    g.nodes = {{"c1", h}, {"dup", y}};
    g.outputs = {y};
    return g;
}

// Breaks: only RELU is modelled among the unary ops. This is the ReLU trap's
// other half -- GGML_OP_UNARY with a different tag -- so it must be refused by
// the tag and not by the op name, which is shared with a legal graph.
//
// SILU and not, say, ggml_sqr: sqr is a GGML_OP_SQR of its own in this ggml and
// never reaches the unary branch at all, so it would test the wrong rule while
// looking like it tested this one. SILU goes through ggml_unary() like relu does
// and is the activation an LLM FFN actually uses.
Graph make_reject_unary(ggml_context * ctx) {
    ggml_tensor * W1 = make(ctx, "W1", 4, 4);
    ggml_tensor * x = make(ctx, "x", 4, 1);
    ggml_tensor * h = ggml_mul_mat(ctx, W1, x);
    ggml_set_name(h, "h");
    ggml_tensor * y = ggml_silu(ctx, h);
    ggml_set_name(y, "y");

    Graph g{"reject-unary",
            "negative fixture: GGML_UNARY_OP_SILU is not modelled",
            {},
            {},
            {}};
    g.leaves = {
        {"W1", "weight", "dram", W1},
        {"x", "input", "vram", x},
    };
    g.nodes = {{"c1", h}, {"silu", y}};
    g.outputs = {y};
    return g;
}

// Breaks: no in-place sharing. ggml_add_inplace() returns view(a), so the
// result has op == GGML_OP_ADD -- a fully supported op -- while aliasing its
// input's storage. This is the case the view_src check exists for: the op name
// looks fine and only view_src reveals that one storage would be counted twice.
Graph make_reject_inplace(ggml_context * ctx) {
    ggml_tensor * W1 = make(ctx, "W1", 4, 4);
    ggml_tensor * W2 = make(ctx, "W2", 4, 4);
    ggml_tensor * x = make(ctx, "x", 4, 1);
    ggml_tensor * a = ggml_mul_mat(ctx, W1, x);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_mul_mat(ctx, W2, x);
    ggml_set_name(b, "b");
    ggml_tensor * y = ggml_add_inplace(ctx, a, b);
    ggml_set_name(y, "y");

    Graph g{"reject-inplace",
            "negative fixture: an in-place ADD that aliases its input",
            {},
            {},
            {}};
    g.leaves = {
        {"W1", "weight", "dram", W1},
        {"W2", "weight", "dram", W2},
        {"x", "input", "vram", x},
    };
    g.nodes = {{"c1", a}, {"c2", b}, {"add", y}};
    g.outputs = {y};
    return g;
}

}  // namespace

std::vector<const char *> graph_names() {
    return {"chain", "residual", "fork", "matvec"};
}

std::vector<const char *> rejection_graph_names() {
    return {"reject-f16", "reject-op", "reject-unary", "reject-inplace"};
}

Graph build_graph(ggml_context * ctx, const char * name) {
    const std::string wanted = (name != nullptr) ? name : "";
    if (wanted == "chain") {
        return make_chain(ctx);
    }
    if (wanted == "residual") {
        return make_residual(ctx);
    }
    if (wanted == "fork") {
        return make_fork(ctx);
    }
    if (wanted == "matvec") {
        return make_matvec(ctx);
    }
    if (wanted == "reject-f16") {
        return make_reject_f16(ctx);
    }
    if (wanted == "reject-op") {
        return make_reject_op(ctx);
    }
    if (wanted == "reject-unary") {
        return make_reject_unary(ctx);
    }
    if (wanted == "reject-inplace") {
        return make_reject_inplace(ctx);
    }
    // Unknown name: an id-less Graph, which callers must treat as a refusal.
    return Graph{nullptr, nullptr, {}, {}, {}};
}

}  // namespace ggml_export
