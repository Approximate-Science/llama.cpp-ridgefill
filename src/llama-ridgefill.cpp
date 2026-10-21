#include "llama-ridgefill.h"

#include "llama-arch.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-vocab.h"
#include "models/models.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>

static const char * LLAMA_RIDGEFILL_KV_SPLIT   = "adapter.ridgefill.split";
static const char * LLAMA_RIDGEFILL_KV_N_LAYER = "adapter.ridgefill.n_layer";
static const char * LLAMA_RIDGEFILL_KV_N_EMBD  = "adapter.ridgefill.n_embd";
static const char * LLAMA_RIDGEFILL_KV_FORMAT  = "adapter.ridgefill.format";

static const char * LLAMA_RIDGEFILL_MODEL_META = "adapter.ridgefill.model_meta.";

static int64_t ridgefill_find(const gguf_context * ctx, const char * key, gguf_type type, bool required) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        if (required) {
            throw std::runtime_error(format("missing key '%s'", key));
        }
        return -1;
    }
    if (gguf_get_kv_type(ctx, id) != type) {
        throw std::runtime_error(format("key '%s' has type %s, expected %s", key,
                gguf_type_name(gguf_get_kv_type(ctx, id)), gguf_type_name(type)));
    }
    return id;
}

static uint32_t ridgefill_get_u32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_u32(ctx, ridgefill_find(ctx, key, GGUF_TYPE_UINT32, true));
}

static float ridgefill_get_f32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_f32(ctx, ridgefill_find(ctx, key, GGUF_TYPE_FLOAT32, true));
}

static std::string ridgefill_get_str(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return "";
    }
    return gguf_get_val_str(ctx, id);
}

static std::string ridgefill_require_str(const gguf_context * ctx, const char * key) {
    return gguf_get_val_str(ctx, ridgefill_find(ctx, key, GGUF_TYPE_STRING, true));
}

//
// SHA-256 (FIPS 180-4), for the vocabulary fingerprint
//

namespace {
struct ridgefill_sha256 {
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    uint8_t  buf[64];
    size_t   n_buf = 0;
    uint64_t n_bits = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t) p[4*i] << 24 | (uint32_t) p[4*i + 1] << 16 | (uint32_t) p[4*i + 2] << 8 | p[4*i + 3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19)  ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const void * data, size_t n) {
        const uint8_t * p = (const uint8_t *) data;
        n_bits += (uint64_t) n * 8;
        while (n > 0) {
            const size_t m = std::min(n, sizeof(buf) - n_buf);
            memcpy(buf + n_buf, p, m);
            n_buf += m; p += m; n -= m;
            if (n_buf == sizeof(buf)) {
                block(buf);
                n_buf = 0;
            }
        }
    }

    std::string hex() {
        const uint64_t bits = n_bits;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (n_buf != 56) {
            update(&zero, 1);
        }
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) {
            len[i] = (uint8_t) (bits >> (56 - 8*i));
        }
        update(len, 8);
        std::string out;
        for (uint32_t x : h) {
            out += format("%08x", x);
        }
        return out;
    }
};
}

// sha256 over the token texts in id order, each followed by one NUL byte (RF27-INTERFACE.md §1)
static std::string ridgefill_vocab_sha256(const llama_vocab & vocab) {
    ridgefill_sha256 sha;
    const uint8_t nul = 0;
    for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
        const char * text = vocab.token_get_text(id);
        sha.update(text, strlen(text));
        sha.update(&nul, 1);
    }
    return sha.hex();
}

// adapter.ridgefill.model_meta.<key>: every key is compared with the model, an unknown key is refused
static void ridgefill_check_model_meta(const llama_model & model, const gguf_context * gctx) {
    const auto & hp = model.hparams;
    const int n_layer = hp.n_layer();

    int il_attn = -1;
    int il_recr = -1;
    for (int il = 0; il < n_layer; ++il) {
        if (hp.is_recr(il)) {
            il_recr = il_recr < 0 ? il : il_recr;
        } else {
            il_attn = il_attn < 0 ? il : il_attn;
        }
    }

    const size_t n_prefix = strlen(LLAMA_RIDGEFILL_MODEL_META);
    int n_checked = 0;

    for (int64_t id = 0; id < gguf_get_n_kv(gctx); ++id) {
        const std::string key = gguf_get_key(gctx, id);
        if (key.compare(0, n_prefix, LLAMA_RIDGEFILL_MODEL_META) != 0) {
            continue;
        }
        const std::string name = key.substr(n_prefix);
        if (gguf_get_kv_type(gctx, id) != GGUF_TYPE_UINT32) {
            throw std::runtime_error(format("projector model_meta '%s' has type %s, expected uint32", name.c_str(),
                    gguf_type_name(gguf_get_kv_type(gctx, id))));
        }
        const int64_t value = gguf_get_val_u32(gctx, id);

        int64_t expect = -1;
        if (name == "num_hidden_layers") {
            expect = n_layer;
        } else if (name == "hidden_size") {
            expect = hp.n_embd;
        } else if (name == "intermediate_size") {
            expect = hp.n_ff(0);
            for (int il = 1; il < n_layer; ++il) {
                if ((int64_t) hp.n_ff(il) != expect) {
                    throw std::runtime_error("model_meta intermediate_size: the model's layers differ in n_ff");
                }
            }
        } else if (name == "vocab_size") {
            expect = model.vocab.n_tokens();
        } else if (name == "num_attention_heads" && il_attn >= 0) {
            expect = hp.n_head(il_attn);
        } else if (name == "num_key_value_heads" && il_attn >= 0) {
            expect = hp.n_head_kv(il_attn);
        } else if (name == "head_dim") {
            expect = hp.n_embd_head_k();
        } else if (name == "full_attention_interval" && il_recr >= 0) {
            expect = value;
            for (int il = 0; il < n_layer; ++il) {
                if (value == 0 || hp.is_recr(il) != ((il + 1) % value != 0)) {
                    expect = -1;
                    break;
                }
            }
        } else if (name == "linear_num_key_heads" && il_recr >= 0) {
            expect = hp.ssm_n_group;
        } else if (name == "linear_num_value_heads" && il_recr >= 0) {
            expect = hp.ssm_dt_rank;
        } else if (name == "linear_key_head_dim" && il_recr >= 0) {
            expect = hp.ssm_d_state;
        } else if (name == "linear_value_head_dim" && il_recr >= 0) {
            expect = hp.ssm_dt_rank > 0 ? hp.ssm_d_inner / hp.ssm_dt_rank : -1;
        } else {
            throw std::runtime_error(format("projector model_meta '%s' is unknown or does not apply to this model", name.c_str()));
        }
        if (value != expect) {
            throw std::runtime_error(format("projector model_meta '%s' = %lld, the model has %lld", name.c_str(),
                    (long long) value, (long long) expect));
        }
        ++n_checked;
    }

    if (n_checked == 0) {
        throw std::runtime_error("projector has no adapter.ridgefill.model_meta.* entries");
    }
}

static void llama_ridgefill_init_impl(const llama_model & model, const char * path, llama_ridgefill & rf) {
    const auto & hparams = model.hparams;

    const int     n_layer = hparams.n_layer();
    const int64_t n_embd  = hparams.n_embd;

    const bool hybrid = model.arch == LLM_ARCH_QWEN35;

    // the fill graph knows these blocks only; qwen35's text positions are equal across its M-RoPE sections
    if ((model.arch != LLM_ARCH_LLAMA && model.arch != LLM_ARCH_QWEN3 && !hybrid) ||
        hparams.n_expert > 0 || hparams.use_kq_norm || hparams.swa_type != LLAMA_SWA_TYPE_NONE ||
        (!hybrid && hparams.n_pos_per_embd() != 1)) {
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

    {
        const int64_t id = ridgefill_find(gctx, LLAMA_RIDGEFILL_KV_FORMAT, GGUF_TYPE_UINT32, false);
        rf.format = id < 0 ? 1 : (int32_t) gguf_get_val_u32(gctx, id);
    }
    if (rf.format != 1 && rf.format != 2) {
        throw std::runtime_error(format("projector format %d is not supported (this build reads 1 and 2)", rf.format));
    }
    if (hybrid && rf.format != 2) {
        throw std::runtime_error(format("a '%s' projector must be format 2 (adapter.ridgefill.format), got %d", arch.c_str(), rf.format));
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

    if (rf.format == 2) {
        ridgefill_check_model_meta(model, gctx);

        const std::string vocab_sha = ridgefill_require_str(gctx, "adapter.ridgefill.vocab_sha256");
        const std::string model_sha = ridgefill_vocab_sha256(model.vocab);
        if (vocab_sha != model_sha) {
            throw std::runtime_error(format("projector vocab_sha256 %s does not match the model's vocabulary %s (another tokenizer)",
                    vocab_sha.c_str(), model_sha.c_str()));
        }

        {
            const int64_t id = ridgefill_find(gctx, "adapter.ridgefill.model_layer_types", GGUF_TYPE_STRING, false);
            if (id >= 0) {
                std::string expect;
                for (int il = 0; il < n_layer; ++il) {
                    expect += il ? " " : "";
                    expect += hparams.is_recr(il) ? "linear_attention" : "full_attention";
                }
                if (expect != gguf_get_val_str(gctx, id)) {
                    throw std::runtime_error("projector model_layer_types do not match the model's layer kinds");
                }
            }
        }

        rf.alpha_default  = ridgefill_get_f32(gctx, "adapter.ridgefill.alpha_default");
        rf.tail_default   = ridgefill_get_u32(gctx, "adapter.ridgefill.tail_default");
        rf.tail_min       = ridgefill_get_u32(gctx, "adapter.ridgefill.tail_min");
        rf.share_default  = ridgefill_get_f32(gctx, "adapter.ridgefill.share_default");
        rf.qattn_default  = ridgefill_get_f32(gctx, "adapter.ridgefill.qattn_default");
        rf.digits_default = gguf_get_val_bool(gctx, ridgefill_find(gctx, "adapter.ridgefill.digits_default", GGUF_TYPE_BOOL, true));

        if (!(rf.alpha_default >= 0.0f && rf.alpha_default <= 1.0f) || rf.tail_min < 1 || rf.tail_default < rf.tail_min ||
            !(rf.share_default > 0.0f && rf.share_default <= 1.0f) || !(rf.qattn_default >= 0.0f && rf.qattn_default <= 1.0f)) {
            throw std::runtime_error("projector defaults (alpha, tail, share, qattn) are out of range");
        }

        std::string authors;
        {
            const int64_t id = ridgefill_find(gctx, "adapter.ridgefill.authors", GGUF_TYPE_ARRAY, true);
            if (gguf_get_arr_type(gctx, id) != GGUF_TYPE_STRING || gguf_get_arr_n(gctx, id) == 0) {
                throw std::runtime_error("adapter.ridgefill.authors must be a non-empty string array");
            }
            for (size_t i = 0; i < gguf_get_arr_n(gctx, id); ++i) {
                authors += (i ? ", " : "") + std::string(gguf_get_arr_str(gctx, id, i));
            }
        }
        rf.desc = format("%s (%s; %s; doi %s)",
                ridgefill_require_str(gctx, "adapter.ridgefill.name").c_str(), authors.c_str(),
                ridgefill_require_str(gctx, "adapter.ridgefill.license").c_str(),
                ridgefill_require_str(gctx, "adapter.ridgefill.doi").c_str());
    } else {
        rf.desc = ridgefill_get_str(gctx, "adapter.ridgefill.source_file");
    }

    // per layer: the map (weight + bias) and, for late GDN layers, the correction st; the row score table
    const int64_t head_v_dim = hybrid && hparams.ssm_dt_rank > 0 ? hparams.ssm_d_inner / hparams.ssm_dt_rank : 0;
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    if (hybrid && (int64_t) hparams.ssm_d_state != head_v_dim) {
        throw std::runtime_error("GDN key and value head sizes differ; the state correction assumes them equal");
    }

    std::vector<ggml_tensor *> w_meta (n_layer, nullptr);
    std::vector<ggml_tensor *> b_meta (n_layer, nullptr);
    std::vector<ggml_tensor *> st_meta(n_layer, nullptr);
    std::set<std::string> expected;

    rf.gdn_idx.assign(n_layer, -1);

    for (int il = split; il < n_layer; ++il) {
        const bool recr = hybrid && hparams.is_recr(il);

        const std::string name_w  = format("blk.%d.ridgefill.weight", il);
        const std::string name_b  = format("blk.%d.ridgefill.bias",   il);
        const std::string name_st = format("blk.%d.ridgefill.st",     il);

        w_meta[il] = ggml_get_tensor(ctx_meta.get(), name_w.c_str());
        b_meta[il] = ggml_get_tensor(ctx_meta.get(), name_b.c_str());

        if (!w_meta[il] && !b_meta[il] && recr) {
            LLAMA_LOG_WARN("%s: no map for GDN layer %d, it is filled with the identity (x = h_S)\n", __func__, il);
        } else if (!w_meta[il] || !b_meta[il]) {
            throw std::runtime_error(format("missing projector tensor '%s'", (!w_meta[il] ? name_w : name_b).c_str()));
        } else {
            if (ggml_n_dims(w_meta[il]) != 2 || w_meta[il]->ne[0] != n_embd || w_meta[il]->ne[1] != n_embd) {
                throw std::runtime_error(format("projector tensor '%s' has a wrong shape", name_w.c_str()));
            }
            if (ggml_n_dims(b_meta[il]) != 1 || b_meta[il]->ne[0] != n_embd || b_meta[il]->type != GGML_TYPE_F32) {
                throw std::runtime_error(format("projector tensor '%s' must be F32 [n_embd]", name_b.c_str()));
            }
            expected.insert(name_w);
            expected.insert(name_b);
        }

        if (recr) {
            if (!model.layers[il].wqkv) {
                throw std::runtime_error(format("GDN layer %d has no fused attn_qkv weight, the fill needs it", il));
            }
            st_meta[il] = ggml_get_tensor(ctx_meta.get(), name_st.c_str());
            if (!st_meta[il]) {
                throw std::runtime_error(format("missing projector tensor '%s'", name_st.c_str()));
            }
            const ggml_tensor * t = st_meta[il];
            if (t->type != GGML_TYPE_F32 || ggml_n_dims(t) != 3 || t->ne[0] != hparams.ssm_d_state || t->ne[1] != head_v_dim || t->ne[2] != n_v_heads) {
                throw std::runtime_error(format("projector tensor '%s' must be F32 [%lld, %lld, %lld]", name_st.c_str(),
                        (long long) hparams.ssm_d_state, (long long) head_v_dim, (long long) n_v_heads));
            }
            expected.insert(name_st);
            rf.gdn_idx[il] = rf.n_gdn_late++;
        }
    }

    ggml_tensor * score_meta = nullptr;
    if (rf.format == 2) {
        score_meta = ggml_get_tensor(ctx_meta.get(), "ridgefill.rowsel.score");
        if (!score_meta || score_meta->type != GGML_TYPE_F32 || ggml_n_dims(score_meta) != 1 ||
            score_meta->ne[0] != (int64_t) model.vocab.n_tokens()) {
            throw std::runtime_error("projector tensor 'ridgefill.rowsel.score' is missing or not F32 [n_vocab]");
        }
        expected.insert("ridgefill.rowsel.score");
    }

    for (int64_t i = 0; i < gguf_get_n_tensors(gctx); ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        if (expected.count(name) == 0) {
            throw std::runtime_error(format("unexpected tensor '%s' in the projector file", name));
        }
    }

    // the map of layer il lives on the device of layer il
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;

    rf.split = split;
    rf.w .assign(n_layer, nullptr);
    rf.b .assign(n_layer, nullptr);
    rf.st.assign(n_layer, nullptr);

    for (int il = split; il < n_layer; ++il) {
        ggml_backend_buffer_type_t buft = model.select_buft(il);

        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ 3*n_layer*ggml_tensor_overhead(),
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

        auto dup = [&](ggml_tensor * meta) -> ggml_tensor * {
            if (!meta) {
                return nullptr;
            }
            ggml_tensor * t = ggml_dup_tensor(it->second, meta);
            ggml_set_name(t, meta->name);
            return t;
        };
        rf.w [il] = dup(w_meta [il]);
        rf.b [il] = dup(b_meta [il]);
        rf.st[il] = dup(st_meta[il]);
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
    auto read_tensor = [&](const char * name, size_t size) -> const uint8_t * {
        const size_t offs = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, gguf_find_tensor(gctx, name));
        if (offs + size < offs || offs + size > file.size()) {
            throw std::runtime_error(format("tensor '%s' data is not within the file bounds", name));
        }
        read_buf.resize(size);
        file.seek(offs, SEEK_SET);
        file.read_raw(read_buf.data(), size);
        return read_buf.data();
    };
    for (int il = split; il < n_layer; ++il) {
        for (ggml_tensor * dst : { rf.w[il], rf.b[il], rf.st[il] }) {
            if (dst) {
                ggml_backend_tensor_set(dst, read_tensor(dst->name, ggml_nbytes(dst)), 0, ggml_nbytes(dst));
            }
        }
    }

    if (score_meta) {
        const size_t n_vocab = model.vocab.n_tokens();
        const float * score = (const float *) read_tensor(score_meta->name, n_vocab*sizeof(float));
        rf.score.assign(score, score + n_vocab);

        // digit rows: the token text contains an ASCII digit; added tokens (control, user-defined) excluded
        rf.is_digit.assign(n_vocab, 0);
        const int excluded = LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_UNUSED;
        for (size_t id = 0; id < n_vocab; ++id) {
            if (model.vocab.token_get_attr(id) & excluded) {
                continue;
            }
            const char * text = model.vocab.token_get_text(id);
            rf.is_digit[id] = strpbrk(text, "0123456789") != nullptr;
        }
    }

    int n_maps = 0;
    for (int il = split; il < n_layer; ++il) {
        n_maps += rf.w[il] != nullptr;
    }
    const ggml_tensor * w_any = nullptr;
    for (int il = split; il < n_layer && !w_any; ++il) {
        w_any = rf.w[il];
    }
    LLAMA_LOG_INFO("%s: loaded RidgeFill projector %s: split = %d, %d maps of type %s, %d state corrections (alpha default %.3g)\n", __func__,
            rf.desc.c_str(), rf.split, n_maps, w_any ? ggml_type_name(w_any->type) : "-", rf.n_gdn_late, rf.alpha_default);
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

float       llama_ridgefill_alpha_default (const llama_ridgefill * rf) { return rf->alpha_default;  }
int32_t     llama_ridgefill_tail_default  (const llama_ridgefill * rf) { return rf->tail_default;   }
int32_t     llama_ridgefill_tail_min      (const llama_ridgefill * rf) { return rf->tail_min;       }
float       llama_ridgefill_share_default (const llama_ridgefill * rf) { return rf->share_default;  }
bool        llama_ridgefill_digits_default(const llama_ridgefill * rf) { return rf->digits_default; }
float       llama_ridgefill_qattn_default (const llama_ridgefill * rf) { return rf->qattn_default;  }
const char * llama_ridgefill_desc         (const llama_ridgefill * rf) { return rf->desc.c_str();   }

void llama_set_ridgefill(llama_context * ctx, const llama_ridgefill * rf) {
    ctx->set_ridgefill(rf);
}

void llama_ridgefill_seq_set(llama_context * ctx, llama_seq_id seq_id, llama_pos p_end) {
    ctx->ridgefill_seq_set(seq_id, p_end);
}

int32_t llama_ridgefill_select_rows(const llama_ridgefill * rf, const llama_token * tokens, int32_t n_prefix,
        float share, bool digits, llama_pos * out) {
    // SPEC §4, the oracle's class_rows (q27/rf27_quality.py) + select.digits, per prompt: of the prefix rows whose
    // token has a finite class score, the round-half-even(share x matches) rarest (highest score), ties by lower
    // position; united with the rows whose token text has an ASCII digit (added tokens excluded); ascending
    if (!rf || rf->score.empty() || rf->is_digit.size() != rf->score.size()) {
        LLAMA_LOG_ERROR("%s: the projector has no row table (format 2 needed)\n", __func__);
        return -1;
    }
    if (n_prefix < 0 || (n_prefix > 0 && (!tokens || !out)) || !(share >= 0.0f && share <= 1.0f)) {
        LLAMA_LOG_ERROR("%s: invalid arguments (n_prefix %d, share %g)\n", __func__, n_prefix, share);
        return -1;
    }

    const int64_t n_vocab = rf->score.size();

    std::vector<int32_t> matches;
    for (int32_t i = 0; i < n_prefix; ++i) {
        if (tokens[i] < 0 || tokens[i] >= n_vocab) {
            LLAMA_LOG_ERROR("%s: token %d at position %d is outside the vocabulary\n", __func__, tokens[i], i);
            return -1;
        }
        if (std::isfinite(rf->score[tokens[i]])) {
            matches.push_back(i);
        }
    }

    // Python round(): half to even, as std::nearbyint in the default rounding mode
    const int64_t k = (int64_t) std::nearbyint((double) share * (double) matches.size());

    std::stable_sort(matches.begin(), matches.end(), [&](int32_t a, int32_t b) {
        return rf->score[tokens[a]] > rf->score[tokens[b]];
    });

    std::vector<uint8_t> pick(n_prefix, 0);
    for (int64_t j = 0; j < k && j < (int64_t) matches.size(); ++j) {
        pick[matches[j]] = 1;
    }
    if (digits) {
        for (int32_t i = 0; i < n_prefix; ++i) {
            pick[i] |= rf->is_digit[tokens[i]];
        }
    }

    int32_t n = 0;
    for (int32_t i = 0; i < n_prefix; ++i) {
        if (pick[i]) {
            out[n++] = i;
        }
    }

    return n;
}

int32_t llama_ridgefill_set_alpha(llama_context * ctx, float alpha) {
    return ctx->ridgefill_set_alpha(alpha);
}

llama_ridgefill_part_params llama_ridgefill_part_default_params(void) {
    llama_ridgefill_part_params p = {};
    p.part = LLAMA_RIDGEFILL_PART_FULL;
    return p;
}

int32_t llama_ridgefill_seq_set_part(llama_context * ctx, llama_seq_id seq_id, const llama_ridgefill_part_params * params) {
    return ctx->ridgefill_seq_set_part(seq_id, params);
}

int32_t llama_ridgefill_seq_set_rows(llama_context * ctx, llama_seq_id seq_id, const llama_pos * rows, int32_t n) {
    return ctx->ridgefill_seq_set_rows(seq_id, rows, n);
}

//
// graph
//

llm_graph_input_ridgefill_part * llm_graph_context::build_inp_ridgefill_part() const {
    const auto & p = ridgefill_params;
    if (!p.rf || p.part == LLM_RIDGEFILL_PART_FULL) {
        return nullptr;
    }
    GGML_ASSERT(p.part_state && p.part_state->h_buf);

    auto inp = std::make_unique<llm_graph_input_ridgefill_part>(p);

    inp->h_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->h_rows);

    if (p.n_win > 0) {
        inp->win_sel  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, p.n_win);
        inp->win_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, p.n_win);
        ggml_set_input(inp->win_sel);
        ggml_set_input(inp->win_rows);
    }

    return (llm_graph_input_ridgefill_part *) res->add_input(std::move(inp));
}

void llm_graph_input_ridgefill_part::set_input(const llama_ubatch * ubatch) {
    const auto * ps = params.part_state;

    // positions (section 0) are the h_buf rows
    std::vector<int32_t> rows(ubatch->n_tokens);
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        rows[i] = ubatch->pos[i];
        GGML_ASSERT(rows[i] >= 0 && rows[i] < ps->h_buf->ne[1]);
    }
    ggml_backend_tensor_set(h_rows, rows.data(), 0, rows.size()*sizeof(int32_t));

    if (win_sel) {
        std::vector<int32_t> sel;
        std::vector<int32_t> dst;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            const llama_pos p = ubatch->pos[i];
            if (p >= ps->win_begin && p < ps->win_begin + ps->win_n) {
                sel.push_back(i);
                dst.push_back(p - ps->win_begin);
            }
        }
        GGML_ASSERT((int64_t) sel.size() == win_sel->ne[0]);
        ggml_backend_tensor_set(win_sel,  sel.data(), 0, sel.size()*sizeof(int32_t));
        ggml_backend_tensor_set(win_rows, dst.data(), 0, dst.size()*sizeof(int32_t));
    }
}

bool llm_graph_input_ridgefill_part::can_reuse(const llm_graph_params & p) {
    return p.ridgefill == params && h_rows->ne[0] == (int64_t) p.ubatch.n_tokens;
}

void build_ridgefill_capture(llm_graph_context & g, const llm_graph_input_ridgefill_part * inp, ggml_tensor * cur, int il) {
    const auto * ps = inp->params.part_state;

    if (il == g.ridgefill->split) {
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx0, ps->h_buf, cur, inp->h_rows));
        return;
    }

    if (!inp->win_sel) {
        return;
    }
    for (size_t k = 0; k < ps->win_layers.size(); ++k) {
        if (ps->win_layers[k] == il) {
            ggml_tensor * rows = ggml_get_rows(g.ctx0, cur, inp->win_sel);
            ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx0, ps->win_bufs[k], rows, inp->win_rows));
        }
    }
}

ggml_tensor * build_ridgefill_h_input(llm_graph_context & g, const llm_graph_input_ridgefill_part * inp) {
    // get_rows returns F32 for F16 / BF16 buffers
    ggml_tensor * cur = ggml_get_rows(g.ctx0, inp->params.part_state->h_buf, inp->h_rows);
    g.cb(cur, "ridgefill_h_S", g.ridgefill->split);
    return cur;
}

int32_t llama_ridgefill_ubatch_rows(const llama_ubatch & ubatch, const std::vector<llama_pos> * rows_seq, std::vector<int32_t> * idx) {
    if (idx) {
        idx->clear();
    }
    if (!rows_seq) {
        return 0;
    }

    int32_t n = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const auto & rows = rows_seq[ubatch.seq_id[i][0]];
        if (!rows.empty() && std::binary_search(rows.begin(), rows.end(), ubatch.pos[i])) {
            if (idx) {
                idx->push_back(i);
            }
            ++n;
        }
    }

    return n;
}

llm_graph_input_ridgefill * llm_graph_context::build_inp_ridgefill() const {
    const auto & p = ridgefill_params;
    if (!p.rf || (p.n_rows == 0 && !p.correct)) {
        return nullptr;
    }

    auto inp = std::make_unique<llm_graph_input_ridgefill>(p);

    if (p.n_rows > 0) {
        inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, p.n_rows);
        ggml_set_input(inp->rows);
        inp->pos  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) p.n_rows*hparams.n_pos_per_embd());
        ggml_set_input(inp->pos);
    }
    if (p.correct) {
        inp->scale = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hparams.ssm_dt_rank, ubatch.n_seqs, p.rf->n_gdn_late);
        ggml_set_input(inp->scale);
    }

    return (llm_graph_input_ridgefill *) res->add_input(std::move(inp));
}

void llm_graph_input_ridgefill::set_input(const llama_ubatch * ubatch) {
    if (rows) {
        std::vector<int32_t> idx;
        llama_ridgefill_ubatch_rows(*ubatch, params.rows_seq, &idx);
        GGML_ASSERT((int64_t) idx.size() == rows->ne[0]);
        GGML_ASSERT(ubatch->token && !ubatch->is_mixed()); // filled ubatches are token-only

        const int64_t n_r   = rows->ne[0];
        const int64_t n_pos = pos->ne[0]/n_r;
        std::vector<int32_t> pos_data(n_r*n_pos);
        for (int64_t k = 0; k < n_pos; ++k) {
            for (int64_t r = 0; r < n_r; ++r) {
                pos_data[k*n_r + r] = ubatch->pos[k*ubatch->n_tokens + idx[r]];
            }
        }
        ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*sizeof(int32_t));
        ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*sizeof(int32_t));
    }

    if (scale) {
        // the correction is added once, on the ubatch that holds the last filled row p_end - 1 of a sequence
        const int64_t n_head = scale->ne[0];
        const int64_t n_seqs = scale->ne[1];
        const int64_t n_lay  = scale->ne[2];
        GGML_ASSERT(n_seqs == ubatch->n_seqs && ubatch->n_tokens == n_seqs*ubatch->n_seq_tokens);

        std::vector<float> data(n_head*n_seqs*n_lay, 0.0f);
        for (int64_t s = 0; s < n_seqs; ++s) {
            const int32_t     i_last = (s + 1)*ubatch->n_seq_tokens - 1;
            const llama_seq_id seq   = ubatch->seq_id[i_last][0];
            if (ubatch->pos[i_last] + 1 != params.p_end[seq]) {
                continue;
            }
            for (int64_t l = 0; l < n_lay; ++l) {
                for (int64_t h = 0; h < n_head; ++h) {
                    data[(l*n_seqs + s)*n_head + h] = params.alpha;
                }
            }
        }
        ggml_backend_tensor_set(scale, data.data(), 0, data.size()*sizeof(float));
    }
}

bool llm_graph_input_ridgefill::can_reuse(const llm_graph_params & p) {
    return p.ridgefill == params &&
        (!scale || (scale->ne[1] == p.ubatch.n_seqs));
}

ggml_tensor * build_ridgefill_late(
        llm_build_delta_net_base        & g,
        const llama_model               & model,
        ggml_tensor                     * h_S,
        const llm_graph_input_ridgefill * rows_mask,
        ggml_tensor                     * inp_pos,
        llm_graph_input_attn_kv         * inp_attn,
        llm_graph_input_rs              * inp_rs,
        llm_ridgefill_fill                mode) {
    const llama_ridgefill * rf = g.ridgefill;
    const auto & hparams = g.hparams;
    ggml_context * ctx0 = g.ctx0;
    ggml_cgraph  * gf   = g.gf;

    GGML_ASSERT(rf && (mode == LLM_RIDGEFILL_FILL_BULK || mode == LLM_RIDGEFILL_FILL_MASKED));
    GGML_ASSERT(g.ubatch.equal_seqs());

    // MASKED: the exact rows R (oracle layer_rows): their residual h_R is carried through every late layer
    const bool masked = mode == LLM_RIDGEFILL_FILL_MASKED;
    GGML_ASSERT(!masked || (rows_mask && rows_mask->rows && rows_mask->pos));

    ggml_tensor * rows  = masked ? rows_mask->rows : nullptr;
    ggml_tensor * h_R   = masked ? ggml_get_rows(ctx0, h_S, rows) : nullptr; // [n_embd, n_R]
    const int64_t n_R   = masked ? rows->ne[0] : 0;

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(hparams.n_embd_head_v())) : hparams.f_attention_scale;

    const int64_t n_tokens     = g.n_tokens;
    const int64_t n_seqs       = g.ubatch.n_seqs;
    const int64_t n_seq_tokens = g.ubatch.n_seq_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v();

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // GDN sizes, as llama_model_qwen35::graph::build_layer_attn_linear
    const int64_t d_inner     = hparams.ssm_d_inner;
    const int64_t head_k_dim  = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim  = d_inner / num_v_heads;

    for (int il = rf->split; il < g.n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * cur = h_S;
        if (rf->w[il]) {
            cur = ggml_mul_mat(ctx0, rf->w[il], h_S);
            cur = ggml_add(ctx0, cur, rf->b[il]);
        }
        g.cb(cur, "ridgefill_inp", il);

        if (masked) {
            // x_mix: the predicted input for bulk rows, the carried exact residual for rows R
            cur = ggml_set_rows(ctx0, cur, h_R, rows);
            g.cb(cur, "ridgefill_mix", il);
        }

        cur = g.build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        g.cb(cur, "attn_norm", il);

        // the normed input of the rows R (Q / z), nullptr in BULK
        ggml_tensor * cur_R = masked ? ggml_get_rows(ctx0, cur, rows) : nullptr;
        ggml_tensor * o_R   = nullptr; // the mixer's output for the rows R, before the residual

        if (!hparams.is_recr(il)) {
            // attention: K, V only (the joint Q/gate projection is not used and so not computed)
            auto [Qcur_full, Kcur, Vcur] = g.build_qkv(layer, cur,
                    n_embd_head * 2, g.n_head,
                    n_embd_head,     g.n_head_kv,
                    n_embd_head,     g.n_head_kv,
                    il, false);
            GGML_UNUSED(Qcur_full);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, g.n_head_kv, n_tokens);
            Kcur = g.build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
            g.cb(Kcur, "Kcur_normed", il);

            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, g.n_head_kv, n_tokens);

            Kcur = ggml_rope_multi(
                    ctx0, Kcur, inp_pos, nullptr,
                    g.n_rot, sections, g.rope_type, g.n_ctx_orig, g.freq_base, g.freq_scale,
                    g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
            g.cb(Kcur, "Kcur", il);
            g.cb(Vcur, "Vcur", il);

            if (inp_attn->self_k_rot) {
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
            }
            if (inp_attn->self_v_rot) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
            }

            const auto * mctx_cur = inp_attn->mctx;
            ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));

            if (masked) {
                // rows R: Q and the output gate, attention over the cache as filled (incl. this ubatch), wo
                auto [Qfull_R, K_R, V_R] = g.build_qkv(layer, cur_R,
                        n_embd_head * 2, g.n_head,
                        n_embd_head,     g.n_head_kv,
                        n_embd_head,     g.n_head_kv,
                        il, false);
                GGML_UNUSED(K_R);
                GGML_UNUSED(V_R);

                ggml_tensor * Q_R = ggml_view_3d(ctx0, Qfull_R, n_embd_head, g.n_head, n_R,
                        ggml_element_size(Qfull_R) * n_embd_head * 2,
                        ggml_element_size(Qfull_R) * n_embd_head * 2 * g.n_head, 0);
                Q_R = g.build_norm(Q_R, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);

                ggml_tensor * gate_R = ggml_view_3d(ctx0, Qfull_R, n_embd_head, g.n_head, n_R,
                        ggml_element_size(Qfull_R) * n_embd_head * 2,
                        ggml_element_size(Qfull_R) * n_embd_head * 2 * g.n_head,
                        ggml_element_size(Qfull_R) * n_embd_head);
                gate_R = ggml_cont_2d(ctx0, gate_R, n_embd_head * g.n_head, n_R);

                Q_R = ggml_rope_multi(
                        ctx0, Q_R, rows_mask->pos, nullptr,
                        g.n_rot, sections, g.rope_type, g.n_ctx_orig, g.freq_base, g.freq_scale,
                        g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
                g.cb(Q_R, "ridgefill_Qcur_R", il);

                if (inp_attn->self_k_rot) {
                    Q_R = llama_mul_mat_hadamard(ctx0, Q_R, inp_attn->self_k_rot);
                }

                // the KQ mask rows of R (one KV stream: refused otherwise in llama_context)
                ggml_tensor * kq_mask = inp_attn->get_kq_mask();
                GGML_ASSERT(kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1);
                ggml_tensor * mask_R = ggml_get_rows(ctx0, kq_mask, rows); // F32 [n_kv, n_R]
                mask_R = ggml_reshape_4d(ctx0, mask_R, mask_R->ne[0], n_R, 1, 1);
                if (g.cparams.flash_attn) {
                    mask_R = ggml_cast(ctx0, mask_R, GGML_TYPE_F16);
                }

                ggml_tensor * k = mctx_cur->get_k(ctx0, il);
                ggml_tensor * v = mctx_cur->get_v(ctx0, il);

                ggml_tensor * o = g.build_attn_mha(Q_R, k, v, nullptr, mask_R, nullptr, nullptr, 0, kq_scale, il);
                if (inp_attn->self_v_rot) {
                    o = llama_mul_mat_hadamard(ctx0, o, inp_attn->self_v_rot);
                }
                o = ggml_mul(ctx0, o, ggml_sigmoid(ctx0, gate_R));
                o_R = g.build_lora_mm(layer.wo, o, layer.wo_s);
                g.cb(o_R, "ridgefill_attn_out_R", il);
            }
        } else {
            // Gated DeltaNet: the conv and recurrent state, as build_layer_attn_linear without z, the gated norm and ssm_out
            const auto * mctx_cur = inp_rs->mctx;

            ggml_tensor * qkv_mixed = g.build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
            qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
            g.cb(qkv_mixed, "linear_attn_qkv_mixed", il);

            ggml_tensor * beta = g.build_lora_mm(layer.ssm_beta, cur, layer.ssm_beta_s);
            beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
            beta = ggml_sigmoid(ctx0, beta);
            g.cb(beta, "beta_sigmoid", il);

            ggml_tensor * alpha = g.build_lora_mm(layer.ssm_alpha, cur, layer.ssm_alpha_s);
            alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);

            ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, layer.ssm_dt);
            ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
            ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, layer.ssm_a); // -A_log.exp() * softplus
            gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);
            g.cb(gate, "gate", il);

            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

            ggml_tensor * conv_kernel      = layer.ssm_conv1d;
            const int64_t conv_kernel_size = conv_kernel->ne[0];
            const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

            ggml_tensor * conv_input = g.build_conv_state(inp_rs, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

            ggml_tensor * state = g.build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);

            ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, conv_kernel));
            g.cb(conv_out, "conv_output_silu", il);

            const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
            const int64_t nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

            ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                    ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
            ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                    ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
                    head_k_dim * num_k_heads * ggml_element_size(conv_out));
            ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_out, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
                    ggml_row_size(conv_out->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
                    ggml_row_size(conv_out->type, 2 * head_k_dim * num_k_heads));

            const float eps_norm = hparams.f_norm_rms_eps;
            q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
            k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

            if (num_k_heads != num_v_heads && (!g.cparams.fused_gdn_ar || !g.cparams.fused_gdn_ch)) {
                GGML_ASSERT(num_v_heads % num_k_heads == 0);
                q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
                k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
            }

            const bool correct = rows_mask && rows_mask->scale;

            ggml_tensor * output = nullptr; // [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
            if (!correct) {
                // stock state write (incl. the rollback snapshots when n_rs_seq > 0); the output is used by rows R only
                output = g.build_recurrent_attn(inp_rs, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);
            } else {
                // the correction ubatch: state += scale * st before the store (scale is 0 for sequences not ending here).
                // Only the current state slot is written: the rows of a filled ubatch are never rolled back to.
                auto [output_c, new_state] = g.build_delta_net(q_conv, k_conv, v_conv, gate, beta, state, il);
                output = output_c;

                const int64_t j = rf->gdn_idx[il];
                GGML_ASSERT(j >= 0);
                ggml_tensor * s = rows_mask->scale;
                ggml_tensor * scale_l = ggml_view_2d(ctx0, s, s->ne[0], s->ne[1], s->nb[1], j*s->nb[2]);
                scale_l = ggml_reshape_4d(ctx0, scale_l, 1, 1, num_v_heads, n_seqs);

                ggml_tensor * st = ggml_repeat_4d(ctx0, rf->st[il], head_v_dim, head_v_dim, num_v_heads, n_seqs);
                new_state = ggml_add(ctx0, new_state, ggml_mul(ctx0, st, scale_l));
                g.cb(new_state, "ridgefill_state_corrected", il);

                const auto kv_head = mctx_cur->get_head();
                ggml_build_forward_expand(gf,
                        ggml_cpy(ctx0, new_state,
                            ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                                kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));
            }

            if (masked) {
                // rows R: the rule's output at R, the gated norm with z, ssm_out
                ggml_tensor * out = ggml_cont(ctx0, output);
                out = ggml_reshape_2d(ctx0, out, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
                out = ggml_get_rows(ctx0, out, rows);
                out = ggml_reshape_3d(ctx0, out, head_v_dim, num_v_heads, n_R);

                ggml_tensor * z = g.build_lora_mm(layer.wqkv_gate, cur_R, layer.wqkv_gate_s);
                z = ggml_reshape_3d(ctx0, z, head_v_dim, num_v_heads, n_R);

                ggml_tensor * y = g.build_norm(out, layer.ssm_norm, nullptr, LLM_NORM_RMS, il);
                y = ggml_mul(ctx0, y, ggml_silu(ctx0, z));
                y = ggml_reshape_2d(ctx0, y, head_v_dim * num_v_heads, n_R);

                o_R = g.build_lora_mm(layer.ssm_out, y, layer.ssm_out_s);
                g.cb(o_R, "ridgefill_linear_attn_out_R", il);
            }
        }

        if (masked) {
            // rows R: residual, attn_post_norm, MLP, residual -> the input of the next late layer
            GGML_ASSERT(o_R);
            ggml_tensor * ffn_residual = ggml_add(ctx0, o_R, h_R);

            ggml_tensor * t = g.build_norm(ffn_residual, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
            t = g.build_ffn(t,
                    layer.ffn_up,   nullptr, layer.ffn_up_s,
                    layer.ffn_gate, nullptr, layer.ffn_gate_s,
                    layer.ffn_down, nullptr, layer.ffn_down_s,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);

            h_R = ggml_add(ctx0, t, ffn_residual);
            g.cb(h_R, "ridgefill_l_out_R", il);
        }
    }

    // h_R after the last layer has no consumer in a filled ubatch (no logits, it writes no cache): it is not
    // expanded, so the last layer's attention output, ssm_out and MLP for the rows R are pruned from the graph
    return h_R;
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
