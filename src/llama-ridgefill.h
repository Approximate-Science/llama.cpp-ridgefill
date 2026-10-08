#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <vector>

// RidgeFill: the prompt tokens before the exact tail run layers [0, split) only. For each layer il >= split,
// the input of layer il is predicted from the input of layer split with a linear map, and only the K/V of
// layer il are computed from it and stored in the KV cache.
struct llama_ridgefill {
    int32_t split = -1;

    // per layer, nullptr for il < split
    std::vector<ggml_tensor *> w; // [n_embd, n_embd]
    std::vector<ggml_tensor *> b; // [n_embd], F32

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
};
