#include "llama-ridgefill.h"

#include "llama-arch.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include <cerrno>
#include <cstring>
#include <map>
#include <stdexcept>

static const char * LLAMA_RIDGEFILL_KV_SPLIT   = "adapter.ridgefill.split";
static const char * LLAMA_RIDGEFILL_KV_N_LAYER = "adapter.ridgefill.n_layer";
static const char * LLAMA_RIDGEFILL_KV_N_EMBD  = "adapter.ridgefill.n_embd";

static uint32_t ridgefill_get_u32(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        throw std::runtime_error(format("missing key '%s'", key));
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(format("key '%s' is not a uint32", key));
    }
    return gguf_get_val_u32(ctx, id);
}

static std::string ridgefill_get_str(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return "";
    }
    return gguf_get_val_str(ctx, id);
}

static void llama_ridgefill_init_impl(const llama_model & model, const char * path, llama_ridgefill & rf) {
    const auto & hparams = model.hparams;

    const int     n_layer = hparams.n_layer();
    const int64_t n_embd  = hparams.n_embd;

    // the fill graph knows the attention block of these models only
    if ((model.arch != LLM_ARCH_LLAMA && model.arch != LLM_ARCH_QWEN3) ||
        hparams.n_expert > 0 || hparams.use_kq_norm || hparams.swa_type != LLAMA_SWA_TYPE_NONE || hparams.n_pos_per_embd() != 1) {
        throw std::runtime_error(format("model arch '%s' (or its configuration) is not supported", llm_arch_name(model.arch)));
    }

    ggml_context * ctx_init = nullptr;
    gguf_init_params meta_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx_init,
    };

    gguf_context_ptr ctx_gguf { gguf_init_from_file(path, meta_params) };
    if (!ctx_gguf) {
        throw std::runtime_error("failed to read the gguf file");
    }
    ggml_context_ptr ctx_meta { ctx_init };

    const gguf_context * gctx = ctx_gguf.get();

    LLM_KV llm_kv = LLM_KV(LLM_ARCH_UNKNOWN);

    if (ridgefill_get_str(gctx, llm_kv(LLM_KV_GENERAL_TYPE).c_str()) != "adapter" ||
        ridgefill_get_str(gctx, llm_kv(LLM_KV_ADAPTER_TYPE).c_str()) != "ridgefill") {
        throw std::runtime_error("expect general.type 'adapter' and adapter.type 'ridgefill'");
    }
    const std::string arch = ridgefill_get_str(gctx, llm_kv(LLM_KV_GENERAL_ARCHITECTURE).c_str());
    if (llm_arch_from_string(arch) != model.arch) {
        throw std::runtime_error(format("projector arch '%s' does not match the model arch '%s'", arch.c_str(), llm_arch_name(model.arch)));
    }

    if (ridgefill_get_u32(gctx, LLAMA_RIDGEFILL_KV_N_LAYER) != (uint32_t) n_layer ||
        ridgefill_get_u32(gctx, LLAMA_RIDGEFILL_KV_N_EMBD)  != (uint32_t) n_embd) {
        throw std::runtime_error("projector n_layer or n_embd does not match the model");
    }
    const uint32_t split_u32 = ridgefill_get_u32(gctx, LLAMA_RIDGEFILL_KV_SPLIT);
    if (split_u32 < 1 || split_u32 >= (uint32_t) n_layer) {
        throw std::runtime_error(format("split %u is out of range [1, %d)", split_u32, n_layer));
    }
    const int split = split_u32;

    // one map per layer in [split, n_layer), nothing else
    std::vector<ggml_tensor *> w_meta(n_layer, nullptr);
    std::vector<ggml_tensor *> b_meta(n_layer, nullptr);

    for (int il = split; il < n_layer; ++il) {
        w_meta[il] = ggml_get_tensor(ctx_meta.get(), format("blk.%d.ridgefill.weight", il).c_str());
        b_meta[il] = ggml_get_tensor(ctx_meta.get(), format("blk.%d.ridgefill.bias",   il).c_str());

        if (!w_meta[il] || !b_meta[il]) {
            throw std::runtime_error(format("missing projector tensors for layer %d", il));
        }
        if (ggml_n_dims(w_meta[il]) != 2 || w_meta[il]->ne[0] != n_embd || w_meta[il]->ne[1] != n_embd) {
            throw std::runtime_error(format("projector weight of layer %d has a wrong shape", il));
        }
        if (ggml_n_dims(b_meta[il]) != 1 || b_meta[il]->ne[0] != n_embd || b_meta[il]->type != GGML_TYPE_F32) {
            throw std::runtime_error(format("projector bias of layer %d must be F32 [n_embd]", il));
        }
    }
    if (gguf_get_n_tensors(gctx) != 2*(int64_t) (n_layer - split)) {
        throw std::runtime_error("unexpected tensors in the projector file");
    }

    // the map of layer il lives on the device of layer il
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;

    rf.split = split;
    rf.w.assign(n_layer, nullptr);
    rf.b.assign(n_layer, nullptr);

    for (int il = split; il < n_layer; ++il) {
        ggml_backend_buffer_type_t buft = model.select_buft(il);

        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*n_layer*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create a ggml context");
            }
            rf.ctxs.emplace_back(ctx);
            it = ctx_map.emplace(buft, ctx).first;
        }

        rf.w[il] = ggml_dup_tensor(it->second, w_meta[il]);
        rf.b[il] = ggml_dup_tensor(it->second, b_meta[il]);
        ggml_set_name(rf.w[il], w_meta[il]->name);
        ggml_set_name(rf.b[il], b_meta[il]->name);
    }

    for (auto & it : ctx_map) {
        ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(it.second, it.first) };
        if (!buf) {
            throw std::runtime_error("failed to allocate the projector buffer");
        }
        LLAMA_LOG_INFO("%s: %10s RidgeFill buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
        rf.bufs.emplace_back(std::move(buf));
    }

    llama_file file(path, "rb");
    std::vector<uint8_t> read_buf;
    for (int il = split; il < n_layer; ++il) {
        for (ggml_tensor * dst : { rf.w[il], rf.b[il] }) {
            const size_t offs = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, gguf_find_tensor(gctx, dst->name));
            const size_t size = ggml_nbytes(dst);
            if (offs + size < offs || offs + size > file.size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds", dst->name));
            }
            read_buf.resize(size);
            file.seek(offs, SEEK_SET);
            file.read_raw(read_buf.data(), size);
            ggml_backend_tensor_set(dst, read_buf.data(), 0, size);
        }
    }

    LLAMA_LOG_INFO("%s: loaded RidgeFill projector, split = %d, %d maps of type %s\n", __func__,
            rf.split, n_layer - split, ggml_type_name(rf.w[split]->type));
}

llama_ridgefill * llama_ridgefill_init(const llama_model * model, const char * path) {
    LLAMA_LOG_INFO("%s: loading RidgeFill projector from '%s' ...\n", __func__, path);

    llama_ridgefill * rf = new llama_ridgefill();

    try {
        llama_ridgefill_init_impl(*model, path, *rf);
        return rf;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to load RidgeFill projector '%s': %s\n", __func__, path, err.what());
    }

    delete rf;

    return nullptr;
}

void llama_ridgefill_free(llama_ridgefill * rf) {
    delete rf;
}

int32_t llama_ridgefill_split(const llama_ridgefill * rf) {
    return rf->split;
}

void llama_set_ridgefill(llama_context * ctx, const llama_ridgefill * rf) {
    ctx->set_ridgefill(rf);
}

void llama_ridgefill_seq_set(llama_context * ctx, llama_seq_id seq_id, llama_pos p_end) {
    ctx->ridgefill_seq_set(seq_id, p_end);
}

bool llm_graph_context::build_ridgefill(
        const llama_model & model,
                      int   il,
              ggml_tensor * inp,
              ggml_tensor * inp_pos,
  llm_graph_input_attn_kv * inp_attn,
              ggml_tensor * inp_out_ids) const {
    if (!ridgefill || il != ridgefill->split) {
        return false;
    }

    const auto * mctx_cur = inp_attn->mctx;

    for (int il_fill = il; il_fill < n_layer; ++il_fill) {
        const auto & layer = model.layers[il_fill];

        ggml_tensor * cur = ggml_mul_mat(ctx0, ridgefill->w[il_fill], inp);
        cur = ggml_add(ctx0, cur, ridgefill->b[il_fill]);
        cb(cur, "ridgefill_inp", il_fill);

        cur = build_norm(cur, layer.attn_norm, NULL, LLM_NORM_RMS, il_fill);
        cb(cur, "attn_norm", il_fill);

        // the Q projection is not used and so not computed
        auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head_k, n_head, n_head_kv, il_fill);
        GGML_UNUSED(Qcur);

        if (layer.attn_k_norm) {
            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il_fill);
            cb(Kcur, "Kcur_normed", il_fill);
        }

        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, model.get_rope_factors(cparams, il_fill),
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        cb(Kcur, "Kcur", il_fill);

        if (inp_attn->self_k_rot) {
            Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
        }
        if (inp_attn->self_v_rot) {
            Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
        }

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il_fill));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il_fill));
    }

    // a filled ubatch has no outputs, but set_inputs() still writes this input
    if (inp_out_ids) {
        ggml_build_forward_expand(gf, inp_out_ids);
    }

    return true;
}
