#include "llama-ridgefill.h"

#include "llama-arch.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-mmap.h"
#include "llama-ridgefill-sweep.h"
#include "llama-model.h"
#include "llama-vocab.h"
#include "models/models.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cinttypes>
#include <map>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>

static const char * LLAMA_RIDGEFILL_KV_SPLIT   = "adapter.ridgefill.split";

struct llama_ridgefill_mask_cache {
    struct entry { std::vector<llama_pos> queries; std::vector<uint8_t> data; };
    std::vector<llama_pos> cells;
    std::map<llama_pos, entry> blocks;
    ggml_type type = GGML_TYPE_COUNT;
    size_t bytes = 0;
    uint64_t hits = 0, misses = 0, invalidations = 0;
    ~llama_ridgefill_mask_cache() {
        LLAMA_LOG_INFO("RidgeFill mask cache: hits %" PRIu64 " misses %" PRIu64 " invalidations %" PRIu64 " bytes %zu\n", hits, misses, invalidations, bytes);
    }
};
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
            !(rf.share_default >= 0.0f && rf.share_default <= 1.0f) || !(rf.qattn_default >= 0.0f && rf.qattn_default <= 1.0f)) {
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

    // Streamed maps and unused alpha-zero corrections stay in host RAM.
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;

    rf.split = split;
    rf.w .assign(n_layer, nullptr);
    rf.b .assign(n_layer, nullptr);
    rf.st.assign(n_layer, nullptr);

    auto dup = [&](ggml_tensor * meta, ggml_backend_buffer_type_t buft) -> ggml_tensor * {
        if (!meta) { return nullptr; }
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

        ggml_tensor * t = ggml_dup_tensor(it->second, meta);
        ggml_set_name(t, meta->name);
        return t;
    };
    for (int il = split; il < n_layer; ++il) {
        const auto buft = model.select_buft(il);
        const auto maps = rf.streamed ? ggml_backend_cpu_buffer_type() : buft;
        rf.w [il] = dup(w_meta [il], maps);
        rf.b [il] = dup(b_meta [il], maps);
        rf.st[il] = dup(st_meta[il], rf.streamed ? maps : buft);
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

        // digit rows: the token text contains an ASCII digit; added tokens (control, user-defined) excluded.
        // special rows: the added tokens themselves (llama.cpp's converter stores HF added_tokens_decoder entries as
        // control (special) or user-defined (non-special) tokens, <unk> as unknown); padding (unused) is neither
        rf.is_digit.assign(n_vocab, 0);
        rf.is_special.assign(n_vocab, 0);
        const int excluded = LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_UNUSED;
        const int special  = LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN;
        for (size_t id = 0; id < n_vocab; ++id) {
            rf.is_special[id] = (model.vocab.token_get_attr(id) & special) != 0;
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

static llama_ridgefill * ridgefill_load(const llama_model * model, const char * path, bool streamed) {
    LLAMA_LOG_INFO("%s: loading RidgeFill projector from '%s' ...\n", __func__, path);

    llama_ridgefill * rf = new llama_ridgefill();
    rf->streamed = streamed;
    rf->profiling = std::getenv("LLAMA_RIDGEFILL_PROFILE") != nullptr;
    rf->trim_kv = std::getenv("LLAMA_RIDGEFILL_TRIM_KV") != nullptr;

    try {
        if (streamed && model->arch != LLM_ARCH_QWEN35) {
            throw std::runtime_error("streamed projector requires the qwen35 layer-major server/scorer sweep");
        }
        llama_ridgefill_init_impl(*model, path, *rf);
        return rf;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to load RidgeFill projector '%s': %s\n", __func__, path, err.what());
    }

    delete rf;

    return nullptr;
}

llama_ridgefill * llama_ridgefill_init(const llama_model * model, const char * path) {
    return ridgefill_load(model, path, false);
}

llama_ridgefill * llama_ridgefill_init_streamed(const llama_model * model, const char * path) {
    return ridgefill_load(model, path, true);
}

bool llama_ridgefill_is_streamed(const llama_ridgefill * rf) { return rf && rf->streamed; }

void llama_ridgefill_profile(const llama_ridgefill * rf, const char * phase, int32_t layer, int32_t pos, int64_t begin, int64_t end) {
    if (rf && rf->profiling) {
        LLAMA_LOG_INFO("RF_PROFILE phase=%s layer=%d pos=%d begin_us=%" PRId64 " end_us=%" PRId64 "\n", phase, layer, pos, begin, end);
    }
}

llama_ridgefill_stream::llama_ridgefill_stream(const llama_model & model, const llama_ridgefill & rf) : rf(rf) {
    const size_t n = rf.w.size();
    if (!rf.streamed || n > SIZE_MAX/(4*ggml_tensor_overhead())) {
        throw std::runtime_error("RidgeFill stream: invalid projector metadata size");
    }
    ggml_init_params params = {4*n*ggml_tensor_overhead(), nullptr, true};
    metadata.reset(ggml_init(params));
    if (!metadata) { throw std::runtime_error("RidgeFill stream: metadata allocation failed"); }
    device.assign(n, -1);
    selected.assign(n, -1);
    w.resize(n);
    b.resize(n);
    std::vector<size_t> matrix_bytes, bias_bytes;
    auto aligned = [](size_t bytes, size_t alignment) {
        if (!alignment || bytes > SIZE_MAX - (alignment - 1)) {
            throw std::runtime_error("RidgeFill stream: map alignment overflow");
        }
        return (bytes + alignment - 1)/alignment*alignment;
    };
    for (size_t il = rf.split; il < n; ++il) {
        if (!rf.w[il]) { continue; }
        if (!ggml_backend_buffer_is_host(rf.w[il]->buffer) || !ggml_backend_buffer_is_host(rf.b[il]->buffer)) {
            throw std::runtime_error("RidgeFill stream: projector maps must be host-resident");
        }
        const auto buft = model.select_buft(il);
        size_t d = 0;
        while (d < devices.size() && devices[d].buft != buft) { ++d; }
        if (d == devices.size()) {
            devices.emplace_back();
            devices.back().buft = buft;
            matrix_bytes.push_back(0);
            bias_bytes.push_back(0);
        }
        device[il] = d;
        matrix_bytes[d] = std::max(matrix_bytes[d], aligned(ggml_backend_buft_get_alloc_size(buft, rf.w[il]), ggml_backend_buft_get_alignment(buft)));
        bias_bytes[d] = std::max(bias_bytes[d], ggml_backend_buft_get_alloc_size(buft, rf.b[il]));
        for (int slot = 0; slot < 2; ++slot) {
            w[il][slot] = ggml_dup_tensor(metadata.get(), rf.w[il]);
            b[il][slot] = ggml_dup_tensor(metadata.get(), rf.b[il]);
            ggml_format_name(w[il][slot], "ridgefill_stream_w_%zu_%d", il, slot);
            ggml_format_name(b[il][slot], "ridgefill_stream_b_%zu_%d", il, slot);
        }
    }
    for (size_t d = 0; d < devices.size(); ++d) {
        if (matrix_bytes[d] > SIZE_MAX - bias_bytes[d]) { throw std::runtime_error("RidgeFill stream: map buffer size overflow"); }
        const size_t size = matrix_bytes[d] + bias_bytes[d];
        for (int slot = 0; slot < 2; ++slot) {
            auto & buffer = devices[d].buffers[slot];
            buffer.reset(ggml_backend_buft_alloc_buffer(devices[d].buft, size));
            if (!buffer) { throw std::runtime_error("RidgeFill stream: staging allocation failed"); }
            ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            const size_t allocated = ggml_backend_buffer_get_size(buffer.get());
            if (n_bytes > SIZE_MAX - allocated) { throw std::runtime_error("RidgeFill stream: total buffer size overflow"); }
            n_bytes += allocated;
            auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer.get()));
            for (size_t il = rf.split; il < n; ++il) {
                if (device[il] != int32_t(d)) { continue; }
                if (ggml_backend_tensor_alloc(buffer.get(), w[il][slot], base) != GGML_STATUS_SUCCESS ||
                        ggml_backend_tensor_alloc(buffer.get(), b[il][slot], base + matrix_bytes[d]) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("RidgeFill stream: map tensor allocation failed");
                }
            }
        }
        LLAMA_LOG_INFO("%s: RidgeFill two-map %s buffers %.4f MiB\n", __func__, ggml_backend_buft_name(devices[d].buft),
                2.0*size/(1024.0*1024.0));
    }
}

void llama_ridgefill_stream::copy(llama_context & ctx, int32_t layer, int32_t slot, bool prefetch) {
    const int64_t started = rf.profiling ? ggml_time_us() : 0;
    auto & maps = devices.at(device.at(layer));
    if (maps.pending[slot].valid()) { maps.pending[slot].get(); }
    if (prefetch && !ggml_backend_buffer_is_host(maps.buffers[slot].get())) {
        const auto * projector = &rf;
        auto * dst_w = w[layer][slot]; auto * dst_b = b[layer][slot];
        auto * src_w = rf.w[layer]; auto * src_b = rf.b[layer];
        maps.pending[slot] = std::async(std::launch::async, [projector, layer, slot, dst_w, dst_b, src_w, src_b]() {
            const int64_t begin = projector->profiling ? ggml_time_us() : 0;
            ggml_backend_tensor_set(dst_w, src_w->data, 0, ggml_nbytes(src_w));
            ggml_backend_tensor_set(dst_b, src_b->data, 0, ggml_nbytes(src_b));
            if (projector->profiling) { llama_ridgefill_profile(projector, "upload_worker", layer, slot, begin, ggml_time_us()); }
        });
        maps.layers[slot] = layer;
        selected[layer] = slot;
        n_uploaded += ggml_nbytes(src_w) + ggml_nbytes(src_b);
        return;
    }
    ggml_backend_t backend = nullptr;
    if (!ggml_backend_buffer_is_host(maps.buffers[slot].get())) {
        for (int i = 0; i < ggml_backend_sched_get_n_backends(ctx.get_sched()); ++i) {
            auto * candidate = ggml_backend_sched_get_backend(ctx.get_sched(), i);
            if (ggml_backend_get_default_buffer_type(candidate) == maps.buft ||
                    (ggml_backend_buft_get_device(maps.buft) && ggml_backend_get_device(candidate) == ggml_backend_buft_get_device(maps.buft))) {
                backend = candidate; break;
            }
        }
        if (!backend) { throw std::runtime_error("RidgeFill stream: layer backend is unavailable"); }
    }
    for (const auto & tensors : {std::make_pair(w[layer][slot], rf.w[layer]), std::make_pair(b[layer][slot], rf.b[layer])}) {
        if (backend) { ggml_backend_tensor_set_async(backend, tensors.first, tensors.second->data, 0, ggml_nbytes(tensors.second)); }
        else { ggml_backend_tensor_set(tensors.first, tensors.second->data, 0, ggml_nbytes(tensors.second)); }
    }
    maps.layers[slot] = layer;
    selected[layer] = slot;
    if (backend) { n_uploaded += ggml_nbytes(rf.w[layer]) + ggml_nbytes(rf.b[layer]); }
    if (rf.profiling) { llama_ridgefill_profile(&rf, "upload", layer, slot, started, ggml_time_us()); }
    LLAMA_LOG_DEBUG("%s: RidgeFill map layer %d slot %d copied %zu bytes\n", __func__, layer, slot,
            ggml_nbytes(rf.w[layer]) + ggml_nbytes(rf.b[layer]));
}

void llama_ridgefill_stream::stage(llama_context & ctx, int32_t layer, int32_t next_layer) {
    if (layer < rf.split || layer >= int32_t(w.size()) || next_layer >= int32_t(w.size())) {
        throw std::runtime_error("RidgeFill stream: layer outside the projector");
    }
    auto resident = [&](int32_t il) {
        if (il < rf.split || !rf.w[il]) { return -1; }
        auto & maps = devices[device[il]];
        for (int slot = 0; slot < 2; ++slot) { if (maps.layers[slot] == il) { return slot; } }
        return -1;
    };
    int current = resident(layer);
    if (current >= 0 && devices[device[layer]].pending[current].valid()) {
        const int64_t started = rf.profiling ? ggml_time_us() : 0;
        devices[device[layer]].pending[current].get();
        if (rf.profiling) { llama_ridgefill_profile(&rf, "upload_wait", layer, current, started, ggml_time_us()); }
    }
    const bool missing = rf.w[layer] && current < 0;
    const bool prefetch = next_layer >= rf.split && next_layer != layer && rf.w[next_layer] && resident(next_layer) < 0;
    if (missing || prefetch) {
        const int64_t started = rf.profiling ? ggml_time_us() : 0;
        ctx.synchronize();
        if (rf.profiling) { llama_ridgefill_profile(&rf, "stage_sync", layer, -1, started, ggml_time_us()); }
    }
    if (missing) {
        auto & maps = devices[device[layer]];
        current = maps.layers[0] == next_layer ? 1 : 0;
        copy(ctx, layer, current, false);
    }
    if (prefetch) {
        const int slot = rf.w[layer] && device[next_layer] == device[layer] ? 1 - current : 0;
        copy(ctx, next_layer, slot, true);
    }
}

ggml_tensor * llama_ridgefill_stream::weight(int32_t layer) const {
    if (!rf.w.at(layer)) { return nullptr; }
    const int slot = selected.at(layer);
    if (slot < 0 || devices.at(device.at(layer)).layers[slot] != layer || devices.at(device.at(layer)).pending[slot].valid()) {
        throw std::runtime_error("RidgeFill stream: requested map was not staged");
    }
    return w[layer][slot];
}

ggml_tensor * llama_ridgefill_stream::bias(int32_t layer) const {
    return weight(layer) ? b[layer][selected[layer]] : nullptr;
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

int32_t llama_set_ridgefill(llama_context * ctx, const llama_ridgefill * rf) {
    return ctx->set_ridgefill(rf);
}

void llama_ridgefill_seq_set(llama_context * ctx, llama_seq_id seq_id, llama_pos p_end) {
    ctx->ridgefill_seq_set(seq_id, p_end);
}

llama_ridgefill_select_params llama_ridgefill_select_default_params(const llama_ridgefill * rf) {
    llama_ridgefill_select_params p = {};
    p.share         = rf ? rf->share_default  : 0.5f;
    p.digits        = rf ? rf->digits_default : false;
    p.special_exact = false;
    p.header_rows   = 0;
    return p;
}

int32_t llama_ridgefill_select_rows(const llama_ridgefill * rf, const llama_token * tokens, int32_t n_prefix,
        float share, bool digits, llama_pos * out) {
    llama_ridgefill_select_params p = llama_ridgefill_select_default_params(rf);
    p.share  = share;
    p.digits = digits;
    return llama_ridgefill_select_rows_ext(rf, tokens, n_prefix, &p, out);
}

int32_t llama_ridgefill_select_rows_ext(const llama_ridgefill * rf, const llama_token * tokens, int32_t n_prefix,
        const llama_ridgefill_select_params * params, llama_pos * out) {
    // SPEC §4, the oracle's class_rows (q27/rf27_quality.py) + select.digits, per prompt: of the prefix rows whose
    // token has a finite class score, the round-half-even(share x matches) rarest (highest score), ties by lower
    // position; united with the rows whose token text has an ASCII digit (added tokens excluded); with
    // special_exact, the rows whose token is an added token (is_special); with header_rows K, the K rows after each
    // added token (positions i+1 .. i+K of an added token at i, within the prefix); ascending
    if (!rf || rf->score.empty() || rf->is_digit.size() != rf->score.size() || rf->is_special.size() != rf->score.size()) {
        LLAMA_LOG_ERROR("%s: the projector has no row table (format 2 needed)\n", __func__);
        return -1;
    }
    if (!params) {
        LLAMA_LOG_ERROR("%s: missing selection parameters\n", __func__);
        return -1;
    }
    const float share = params->share;
    if (n_prefix < 0 || (n_prefix > 0 && (!tokens || !out)) || !(share >= 0.0f && share <= 1.0f) || params->header_rows < 0) {
        LLAMA_LOG_ERROR("%s: invalid arguments (n_prefix %d, share %g, header_rows %d)\n", __func__, n_prefix, share, params->header_rows);
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
    for (int32_t i = 0; i < n_prefix; ++i) {
        if (params->digits) {
            pick[i] |= rf->is_digit[tokens[i]];
        }
        if (rf->is_special[tokens[i]]) {
            pick[i] |= params->special_exact;
            for (int32_t j = i + 1; j <= i + params->header_rows && j < n_prefix; ++j) {
                pick[j] = 1;
            }
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

bool llama_ridgefill_token_is_special(const llama_ridgefill * rf, llama_token id) {
    return rf && id >= 0 && id < (llama_token) rf->is_special.size() && rf->is_special[id];
}

int32_t llama_ridgefill_last_marker(const llama_token * tokens, int32_t n, const llama_token * marker, int32_t n_marker) {
    // the start of the last occurrence of the token sequence `marker` in tokens[0, n), -1 if none
    if (!tokens || !marker || n < 0 || n_marker <= 0) {
        return -1;
    }
    for (int32_t i = n - n_marker; i >= 0; --i) {
        if (std::equal(marker, marker + n_marker, tokens + i)) {
            return i;
        }
    }
    return -1;
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

int32_t llama_ridgefill_api_version(void) {
    return LLAMA_RIDGEFILL_API_VERSION;
}

int32_t llama_ridgefill_seq_get_s1_mass(llama_context * ctx, llama_seq_id seq_id, double * out, int32_t n) {
    return ctx->ridgefill_seq_get_s1_mass(seq_id, out, n);
}

int32_t llama_ridgefill_seq_set_vtail(llama_context * ctx, llama_seq_id seq_id, bool enabled, int32_t rows) {
    return ctx->ridgefill_seq_set_vtail(seq_id, enabled, rows);
}

int32_t llama_ridgefill_seq_get_vtail_scales(llama_context * ctx, llama_seq_id seq_id, float * out, int32_t n) {
    return ctx->ridgefill_seq_get_vtail_scales(seq_id, out, n);
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
    const int64_t carry_count = p.fill ? p.n_rows : n_tokens;
    if (p.stream_layer >= 0 && p.part_state->stream_carry && carry_count > 0) {
        inp->carry_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, carry_count);
        ggml_set_input(inp->carry_rows);
        ggml_build_forward_expand(gf, inp->carry_rows);
    }
    if (p.stream_layer >= 0) { ggml_build_forward_expand(gf, inp->h_rows); }
    if (p.stream_layer >= 0 && !hparams.is_recr(p.stream_layer) && p.part == LLM_RIDGEFILL_PART_LATE &&
            (!p.fill || p.n_rows > 0)) {
        inp->kv = static_cast<const llama_memory_hybrid_context *>(mctx)->get_attn();
        inp->seq = ubatch.seq_id[0][0];
        inp->attention_mask = ggml_new_tensor_4d(ctx0, cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32,
                inp->kv->get_n_kv(), p.fill ? p.n_rows : n_tokens, 1, 1);
        ggml_set_input(inp->attention_mask);
        ggml_set_name(inp->attention_mask, "ridgefill_attention_mask");
    }

    if (p.n_win > 0) {
        inp->win_sel  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, p.n_win);
        inp->win_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, p.n_win);
        ggml_set_input(inp->win_sel);
        ggml_set_input(inp->win_rows);
    }

    if (p.s1_probe || p.vtail_rows > 0) {
        inp->kv  = static_cast<const llama_memory_hybrid_context *>(mctx)->get_attn();
        inp->seq = ubatch.seq_id[0][0];
        int64_t n_kv = inp->kv->get_n_kv();
        if (p.s1_probe) {
            // The reserve state has no prompt. Keep its dense graph as the allocation upper bound.
            if (cparams.kv_unified && cparams.n_seq_max > 1 && p.part_state->lower_end > 0) {
                n_kv = p.part_state->lower_end;
                inp->s1_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
                ggml_set_input(inp->s1_rows);
            }
            // the stock KQ mask's type and shape (build_attn_inp_kq_mask), one sequence / stream
            if (p.certificate && !p.rf->streamed) {
                inp->cert_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
                ggml_set_input(inp->cert_rows);
            }
            inp->s1_mask = ggml_new_tensor_4d(ctx0, cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv, n_tokens, 1, 1);
            ggml_set_input(inp->s1_mask);
            inp->ones = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
            ggml_set_input(inp->ones);
            inp->s1_qb = llama_ridgefill_query_block(n_kv, n_tokens, LLAMA_RIDGEFILL_PROBE_BLOCK);
            for (int64_t q0 = 0; q0 < n_tokens; q0 += inp->s1_qb) {
                const int64_t nq = std::min<int64_t>(inp->s1_qb, n_tokens - q0);
                if (nq == n_tokens) {
                    inp->s1_mask_blk.push_back(inp->s1_mask);
                    inp->s1_ones_blk.push_back(inp->ones);
                } else {
                    inp->s1_mask_blk.push_back(ggml_view_2d(ctx0, inp->s1_mask, n_kv, nq, inp->s1_mask->nb[1], q0*inp->s1_mask->nb[1]));
                    inp->s1_ones_blk.push_back(ggml_view_2d(ctx0, inp->ones, 1, nq, inp->ones->nb[1], 0));
                }
            }
        }
        if (p.vtail_rows > 0) {
            inp->vt_bulk = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_kv);
            inp->vt_keep = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_kv);
            ggml_set_input(inp->vt_bulk);
            ggml_set_input(inp->vt_keep);
        }
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
    if (ps->h_compact) {
        std::vector<int32_t> compact_rows(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            compact_rows[i] = params.stream_host_tail ? rows[i] - ps->h_tail_begin : ps->h_indices->at(rows[i]);
            auto * src = params.stream_host_tail ? ps->h_tail : ps->h_compact;
            GGML_ASSERT(compact_rows[i] >= 0 && compact_rows[i] < src->ne[1]);
        }
        ggml_backend_tensor_set(h_rows, compact_rows.data(), 0, compact_rows.size()*sizeof(int32_t));
    } else { ggml_backend_tensor_set(h_rows, rows.data(), 0, rows.size()*sizeof(int32_t)); }
    if (carry_rows) {
        const auto & exact = *ps->stream_rows;
        const llama_pos prefix = params.p_end[ubatch->seq_id[0][0]];
        std::vector<int32_t> indices;
        for (const auto pos : rows) {
            const auto it = std::lower_bound(exact.begin(), exact.end(), pos);
            if (pos < prefix && (it == exact.end() || *it != pos)) { continue; }
            const int64_t index = pos < prefix ? it - exact.begin() : int64_t(exact.size()) + pos - prefix;
            GGML_ASSERT(index >= 0 && index < ps->stream_carry->ne[1]);
            indices.push_back(index);
        }
        GGML_ASSERT(indices.size() == size_t(carry_rows->ne[0]));
        ggml_backend_tensor_set(carry_rows, indices.data(), 0, indices.size()*sizeof(int32_t));
    }
    if (attention_mask && attention_mask->buffer) {
        std::vector<int32_t> selected;
        if (params.fill) { llama_ridgefill_ubatch_rows(*ubatch, params.rows_seq, &selected); }
        else { for (uint32_t i = 0; i < ubatch->n_tokens; ++i) { selected.push_back(i); } }
        const size_t count = selected.size();
        GGML_ASSERT(count > 0 && count == size_t(attention_mask->ne[1]) && ubatch->token && ubatch->n_seqs_unq == 1);
        std::vector<llama_pos> positions(count*ubatch->n_pos);
        for (uint32_t k = 0; k < ubatch->n_pos; ++k) {
            for (size_t i = 0; i < count; ++i) { positions[k*count + i] = ubatch->pos[size_t(k)*ubatch->n_tokens + selected[i]]; }
        }
        if (!ps->mask_cache) { ps->mask_cache = std::make_shared<llama_ridgefill_mask_cache>(); }
        auto & cache = *ps->mask_cache;
        std::vector<llama_pos> cells;
        kv->get_cell_pos(seq, cells);
        if (cache.cells != cells || cache.type != attention_mask->type) {
            if (!cache.blocks.empty()) { ++cache.invalidations; }
            cache.blocks.clear(); cache.bytes = 0; cache.cells = std::move(cells); cache.type = attention_mask->type;
        }
        auto it = cache.blocks.find(positions[0]);
        const size_t bytes = ggml_nbytes(attention_mask);
        if (it != cache.blocks.end() && it->second.queries == positions && it->second.data.size() == bytes) {
            std::memcpy(attention_mask->data, it->second.data.data(), bytes);
            ++cache.hits;
        } else {
            auto subset = *ubatch;
            subset.n_tokens = subset.n_seq_tokens = count;
            subset.pos = positions.data();
            kv->set_input_kq_mask(attention_mask, &subset, true);
            ++cache.misses;
            if (it != cache.blocks.end()) { cache.bytes -= it->second.data.size(); cache.blocks.erase(it); }
            if (bytes <= LLAMA_RIDGEFILL_SWEEP_MAX_BYTES - cache.bytes) {
                auto & saved = cache.blocks[positions[0]];
                saved.queries = std::move(positions); saved.data.resize(bytes);
                std::memcpy(saved.data.data(), attention_mask->data, bytes); cache.bytes += bytes;
            }
        }
    }

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

    if (s1_mask || vt_bulk) {
        std::vector<llama_pos> cell_pos;
        kv->get_cell_pos(seq, cell_pos);
        const llama_pos p_end = params.p_end[seq];
        if (cert_rows) {
            std::vector<int32_t> cells(ubatch->n_tokens);
            for (uint32_t t = 0; t < ubatch->n_tokens; ++t) {
                auto it = std::find(cell_pos.begin(), cell_pos.end(), ubatch->pos[t]);
                GGML_ASSERT(it != cell_pos.end());
                cells[t] = std::distance(cell_pos.begin(), it);
            }
            ggml_backend_tensor_set(cert_rows, cells.data(), 0, cells.size()*sizeof(int32_t));
        }
        if (s1_mask) {
            // query i sees the sequence's cells at positions <= its own (causal, as the stock mask), except the tail
            // rows [p_end, s1_begin) the probe does not run: the prefix [0, p_end) as filled and the window rows
            const int64_t n_kv = s1_mask->ne[0];
            if (s1_rows) {
                std::vector<int32_t> cells(n_kv, -1);
                for (size_t c = 0; c < cell_pos.size(); ++c) {
                    if (cell_pos[c] >= 0 && cell_pos[c] < n_kv) {
                        if (c > size_t(std::numeric_limits<int32_t>::max())) {
                            throw std::runtime_error("RidgeFill S1: key row exceeds the gather index range");
                        }
                        cells[cell_pos[c]] = int32_t(c);
                    }
                }
                if (std::find(cells.begin(), cells.end(), -1) != cells.end()) {
                    throw std::runtime_error("RidgeFill S1: incomplete sequence key rows");
                }
                ggml_backend_tensor_set(s1_rows, cells.data(), 0, cells.size()*sizeof(int32_t));
                cell_pos.resize(n_kv);
                std::iota(cell_pos.begin(), cell_pos.end(), 0);
            }
            GGML_ASSERT((int64_t) cell_pos.size() == n_kv && s1_mask->ne[1] == ubatch->n_tokens);
            std::vector<float> row(n_kv);
            std::vector<ggml_fp16_t> row16(s1_mask->type == GGML_TYPE_F16 ? n_kv : 0);
            for (uint32_t t = 0; t < ubatch->n_tokens; ++t) {
                const llama_pos pt = ubatch->pos[t];
                for (int64_t c = 0; c < n_kv; ++c) {
                    const llama_pos pc = cell_pos[c];
                    const bool visible = pc >= 0 && pc <= pt && !(pc >= p_end && pc < ps->s1_begin);
                    row[c] = visible ? 0.0f : -INFINITY;
                }
                if (s1_mask->type == GGML_TYPE_F16) {
                    ggml_fp32_to_fp16_row(row.data(), row16.data(), n_kv);
                    ggml_backend_tensor_set(s1_mask, row16.data(), t*s1_mask->nb[1], n_kv*sizeof(ggml_fp16_t));
                } else {
                    ggml_backend_tensor_set(s1_mask, row.data(), t*s1_mask->nb[1], n_kv*sizeof(float));
                }
            }
            const std::vector<float> one(ggml_nelements(ones), 1.0f);
            ggml_backend_tensor_set(ones, one.data(), 0, one.size()*sizeof(float));
        }
        if (vt_bulk) {
            const auto & rows = params.rows_vtail ? *params.rows_vtail : std::vector<llama_pos>();
            std::vector<float> bulk(cell_pos.size(), 0.0f), keep(cell_pos.size(), 1.0f);
            for (size_t i = 0; i < cell_pos.size(); ++i) {
                const llama_pos p = cell_pos[i];
                if (p >= 0 && p < p_end && !std::binary_search(rows.begin(), rows.end(), p)) {
                    bulk[i] = 1.0f;
                    keep[i] = 0.0f;
                }
            }
            ggml_backend_tensor_set(vt_bulk, bulk.data(), 0, bulk.size()*sizeof(float));
            ggml_backend_tensor_set(vt_keep, keep.data(), 0, keep.size()*sizeof(float));
        }
    }
}

bool llm_graph_input_ridgefill_part::can_reuse(const llm_graph_params & p) {
    if (s1_mask || vt_bulk) {
        return false; // sized by the ubatch's n_kv; rare graphs (the probe, the first tail ubatch)
    }
    if (attention_mask) {
        kv = static_cast<const llama_memory_hybrid_context *>(p.mctx)->get_attn();
        if (attention_mask->ne[0] != kv->get_n_kv()) { return false; }
    }
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
    const auto & ps = *inp->params.part_state;
    auto * source = ps.h_compact ? (inp->params.stream_host_tail ? ps.h_tail : ps.h_compact) : ps.h_buf;
    ggml_tensor * cur = ggml_get_rows(g.ctx0, source, inp->h_rows);
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

llm_graph_input_ridgefill * llm_graph_context::build_inp_ridgefill(llm_graph_input_attn_kv * inp_attn) const {
    const auto & p = ridgefill_params;
    if (!p.rf || (p.n_rows == 0 && !p.correct)) {
        return nullptr;
    }

    auto inp = std::make_unique<llm_graph_input_ridgefill>(p);

    if (p.n_rows > 0) {
        inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, p.n_rows);
        ggml_set_input(inp->rows);
        if (p.stream_layer < 0 || !hparams.is_recr(p.stream_layer)) {
            inp->pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) p.n_rows*hparams.n_pos_per_embd());
            ggml_set_input(inp->pos);
        }
        if (p.reuse_kv) {
            inp->attn = inp_attn;
            inp->k_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, p.n_rows);
            inp->v_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, p.n_rows * (inp_attn->get_v_idxs()->ne[0] / n_tokens));
            ggml_set_input(inp->k_idxs);
            ggml_set_input(inp->v_idxs);
            // Keep the host index inputs for the exact-row gather in set_input.
            ggml_build_forward_expand(gf, inp_attn->get_k_idxs());
            ggml_build_forward_expand(gf, inp_attn->get_v_idxs());
        }
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
        const int64_t n_pos = pos ? pos->ne[0]/n_r : 0;
        std::vector<int32_t> pos_data(n_r*n_pos);
        for (int64_t k = 0; k < n_pos; ++k) {
            for (int64_t r = 0; r < n_r; ++r) {
                pos_data[k*n_r + r] = ubatch->pos[k*ubatch->n_tokens + idx[r]];
            }
        }
        if (rows->buffer) { ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*sizeof(int32_t)); }
        if (pos && pos->buffer) { ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*sizeof(int32_t)); }
        if (k_idxs) {
            auto gather = [&](ggml_tensor * src, ggml_tensor * dst) {
                const size_t stride = src->ne[0] / ubatch->n_tokens;
                std::vector<int64_t> full(src->ne[0]), selected(dst->ne[0]);
                ggml_backend_tensor_get(src, full.data(), 0, full.size()*sizeof(int64_t));
                for (size_t r = 0; r < idx.size(); ++r) {
                    std::copy_n(full.data() + idx[r]*stride, stride, selected.data() + r*stride);
                }
                if (dst->buffer) { ggml_backend_tensor_set(dst, selected.data(), 0, selected.size()*sizeof(int64_t)); }
            };
            gather(attn->get_k_idxs(), k_idxs);
            gather(attn->get_v_idxs(), v_idxs);
        }
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
        llm_ridgefill_fill                mode,
        const llm_graph_input_ridgefill_part * part) {
    const llama_ridgefill * rf = g.ridgefill;
    const auto & hparams = g.hparams;
    ggml_context * ctx0 = g.ctx0;
    ggml_cgraph  * gf   = g.gf;

    GGML_ASSERT(rf && (mode == LLM_RIDGEFILL_FILL_BULK || mode == LLM_RIDGEFILL_FILL_MASKED));
    GGML_ASSERT(g.ubatch.equal_seqs());

    // MASKED: the exact rows R (oracle layer_rows): their residual h_R is carried through every late layer
    const bool masked = mode == LLM_RIDGEFILL_FILL_MASKED;
    GGML_ASSERT(!masked || (rows_mask && rows_mask->rows));

    ggml_tensor * rows  = masked ? rows_mask->rows : nullptr;
    ggml_tensor * h_R   = masked ? ggml_get_rows(ctx0, h_S, rows) : nullptr; // [n_embd, n_R]
    const int stream = g.ridgefill_params.stream_layer;
    if (masked && (stream > rf->split || (part && part->params.part_state->h_compact))) {
        h_R = ggml_get_rows(ctx0, part->params.part_state->stream_carry, part->carry_rows);
    }
    const int64_t n_R   = masked ? rows->ne[0] : 0;
    ggml_tensor * mask_R = nullptr; // the KQ mask rows of R, gathered once for all late attention layers

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

    for (int il = stream >= 0 ? stream : rf->split; il < (stream >= 0 ? stream + 1 : g.n_layer); ++il) {
        const auto & layer = model.layers[il];

        if (g.ridgefill_params.s1_attn_only && hparams.is_recr(il)) {
            continue; // S1_FILL: the late GDN layers are not filled (s1_gdn "window")
        }

        const bool reuse_kv = g.ridgefill_params.reuse_kv && !hparams.is_recr(il);
        if (reuse_kv && !masked) { continue; }

        // Keep the original GEMM shape so the selected rows keep the same arithmetic.
        ggml_tensor * cur = reuse_kv ? ggml_set_rows(ctx0, ggml_scale(ctx0, h_S, 0.0f), h_R, rows) : h_S;
        auto * weight = stream >= 0 ? g.ridgefill_params.stream_w : rf->w[il];
        auto * bias = stream >= 0 ? g.ridgefill_params.stream_b : rf->b[il];
        if (!reuse_kv && weight) {
            cur = ggml_mul_mat(ctx0, weight, h_S);
            cur = ggml_add(ctx0, cur, bias);
        }
        g.cb(cur, "ridgefill_inp", il);

        if (masked && !reuse_kv) {
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
            const int64_t n_kv_rows = reuse_kv ? n_R : n_tokens;
            // attention: K, V only (the joint Q/gate projection is not used and so not computed)
            auto [Qcur_full, Kcur, Vcur] = g.build_qkv(layer, cur,
                    n_embd_head * 2, g.n_head,
                    n_embd_head,     g.n_head_kv,
                    n_embd_head,     g.n_head_kv,
                    il, false);
            GGML_UNUSED(Qcur_full);

            // Gather before norm/RoPE to keep the fused cache-store path.
            if (reuse_kv) { Kcur = ggml_get_rows(ctx0, Kcur, rows); }
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, g.n_head_kv, n_kv_rows);
            Kcur = g.build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
            g.cb(Kcur, "Kcur_normed", il);

            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, g.n_head_kv, n_tokens);

            Kcur = ggml_rope_multi(
                    ctx0, Kcur, reuse_kv ? rows_mask->pos : inp_pos, nullptr,
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

            if (reuse_kv) {
                Vcur = ggml_get_rows(ctx0, ggml_cont_2d(ctx0, Vcur, n_embd_head*g.n_head_kv, n_tokens), rows);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, g.n_head_kv, n_R);
            }
            const auto * mctx_cur = inp_attn->mctx;
            ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, reuse_kv ? rows_mask->k_idxs : inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, reuse_kv ? rows_mask->v_idxs : inp_attn->get_v_idxs(), il));

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
                if (!mask_R) {
                    ggml_tensor * kq_mask = part && part->attention_mask ? part->attention_mask : inp_attn->get_kq_mask();
                    GGML_ASSERT(kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1);
                    if (part && part->attention_mask) {
                        // Keep INPUT: a view can defer its host copy until after the next input update.
                        mask_R = kq_mask;
                    } else {
                        mask_R = ggml_get_rows(ctx0, kq_mask, rows); // [n_kv, n_R]
                        mask_R = ggml_reshape_4d(ctx0, mask_R, mask_R->ne[0], n_R, 1, 1);
                        if (g.cparams.flash_attn) {
                            mask_R = ggml_cast(ctx0, mask_R, GGML_TYPE_F16);
                        }
                    }
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

    if (masked && stream >= 0 && stream + 1 < g.n_layer) {
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, part->params.part_state->stream_carry, h_R, part->carry_rows));
    }
    // h_R after the last layer has no consumer in a filled ubatch (no logits, it writes no cache): it is not
    // expanded, so the last layer's attention output, ssm_out and MLP for the rows R are pruned from the graph
    return h_R;
}

void llama_ridgefill_attn_mass_split(int64_t n_kv, int64_t n_q, int64_t rep, int64_t max_elems, int64_t & hb, int64_t & qb) {
    GGML_ASSERT(n_kv > 0 && n_q > 0 && rep > 0 && max_elems > 0);
    if (n_kv*rep*n_q <= max_elems) {
        hb = rep; qb = n_q; // one block per key head: the unsliced summation order
        return;
    }
    hb = 1;
    for (int64_t d = rep; d >= 1; --d) {
        if (rep % d == 0 && n_kv*d <= max_elems) { hb = d; break; }
    }
    qb = std::max<int64_t>(1, std::min<int64_t>(n_q, max_elems/(n_kv*hb)));
}

ggml_tensor * llama_ridgefill_follow(ggml_context * ctx, ggml_tensor * x, ggml_tensor * dep) {
    if (!dep) {
        return x;
    }
    // x + 0 * dep[0]: the same values (dep is finite), but in every topological order x is computed after dep, so a
    // backend that reorders independent nodes (the Vulkan graph optimizer) cannot hoist the next block's scores
    return ggml_add(ctx, x, ggml_scale(ctx, ggml_view_1d(ctx, dep, 1, 0), 0.0f));
}

int64_t llama_ridgefill_query_block(int64_t n_kv, int64_t n_q, int64_t max_elems) {
    if (n_kv*n_q <= max_elems) {
        return n_q;
    }
    const int64_t n_blocks = (n_kv*n_q + max_elems - 1)/max_elems;
    return std::max<int64_t>(1, (n_q + n_blocks - 1)/n_blocks); // equal blocks, not a short last one
}

ggml_tensor * llama_ridgefill_attn_mass(ggml_context * ctx, ggml_tensor * q_base, int64_t h0, int64_t n_heads,
        ggml_tensor * k, ggml_tensor * mask, int32_t diag_past, float scale, ggml_tensor * ones, int64_t max_elems,
        ggml_tensor ** dep) {
    const int64_t d = q_base->ne[0], n_q = q_base->ne[2];
    const int64_t n_kv = k->ne[1], n_head_kv = k->ne[2];
    GGML_ASSERT(k->ne[0] == d && n_heads % n_head_kv == 0 && h0 + n_heads <= q_base->ne[1] && q_base->ne[3] == 1 && k->ne[3] == 1);
    GGML_ASSERT(ggml_is_contiguous(q_base) && dep);
    GGML_ASSERT(!mask || (mask->ne[0] == n_kv && mask->ne[1] >= n_q));
    const int64_t rep = n_heads/n_head_kv;
    int64_t hb, qb;
    llama_ridgefill_attn_mass_split(n_kv, n_q, rep, max_elems, hb, qb);
    GGML_ASSERT(!ones || ones->ne[1] >= hb*qb);

    ggml_tensor * total = nullptr;   // ones: [n_kv]
    ggml_tensor * per_head = nullptr; // sum_rows: [n_kv, n_head_kv]
    for (int64_t g = 0; g < n_head_kv; ++g) {
        ggml_tensor * kg = ggml_view_3d(ctx, k, d, n_kv, 1, k->nb[1], k->nb[2], g*k->nb[2]);
        ggml_tensor * acc = nullptr;
        for (int64_t hl = 0; hl < rep; hl += hb) {
            for (int64_t q0 = 0; q0 < n_q; q0 += qb) {
                const int64_t nq = std::min(qb, n_q - q0);
                // the queries as the unsliced [d, n_q, n_head] permuted view would give them (same strides)
                ggml_tensor * qp = ggml_permute(ctx, llama_ridgefill_follow(ctx, q_base, *dep), 0, 2, 1, 3);
                ggml_tensor * qv = ggml_view_3d(ctx, qp, d, nq, hb, qp->nb[1], qp->nb[2],
                        q0*qp->nb[1] + (h0 + g*rep + hl)*qp->nb[2]);
                ggml_tensor * kq = ggml_mul_mat(ctx, kg, qv); // [n_kv, nq, hb]
                ggml_prec_set_acc(kq, GGML_PREC_F32);
                ggml_tensor * mv = nullptr;
                if (mask) {
                    mv = ggml_view_2d(ctx, mask, n_kv, nq, mask->nb[1], q0*mask->nb[1]);
                } else {
                    kq = ggml_diag_mask_inf(ctx, kq, diag_past + int32_t(q0));
                }
                ggml_tensor * p = ggml_soft_max_ext(ctx, kq, mv, scale, 0.0f);
                ggml_tensor * p2 = ggml_reshape_2d(ctx, p, n_kv, nq*hb);
                ggml_tensor * m;
                if (ones) {
                    // sum over the block's (query, head) columns without a transposed copy
                    m = ggml_out_prod(ctx, p2, ggml_view_2d(ctx, ones, 1, nq*hb, ones->nb[1], 0)); // [n_kv, 1]
                    m = ggml_reshape_1d(ctx, m, n_kv);
                } else {
                    m = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, p2))); // [1, n_kv]
                    m = ggml_reshape_2d(ctx, m, n_kv, 1);
                }
                *dep = m;
                acc = acc ? ggml_add(ctx, acc, m) : m;
            }
        }
        if (ones) {
            total = total ? ggml_add(ctx, total, acc) : acc;
        } else {
            per_head = per_head ? ggml_concat(ctx, per_head, acc, 1) : acc;
        }
    }
    return ones ? total : per_head;
}

void llama_ridgefill_graph_peaks(const ggml_cgraph * gf, size_t & in_order, size_t & any_order, size_t & input_views) {
    const int n = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(gf));
    std::unordered_map<const ggml_tensor *, int> idx;
    for (int i = 0; i < n; ++i) {
        idx[ggml_graph_node(const_cast<ggml_cgraph *>(gf), i)] = i;
    }
    auto node = [&](int i) { return ggml_graph_node(const_cast<ggml_cgraph *>(gf), i); };
    auto root = [](const ggml_tensor * t) { while (t->view_src) { t = t->view_src; } return t; };

    // the earliest step each node can run at (a reorderer that pulls all independent work forward, the worst case for
    // memory): after its producers, and, as reordering backends keep them (the Vulkan graph optimizer: nodes viewing
    // the same tensor), after the earlier writers of any storage it reads (a cache view after the cpy into it) and,
    // for a node writing into a view, after the earlier readers and writers of that storage
    auto writes = [&](const ggml_tensor * t) {
        return t->view_src && t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE &&
               t->op != GGML_OP_TRANSPOSE && t->op != GGML_OP_NONE;
    };
    std::vector<int> level(n, 0);
    std::unordered_map<const ggml_tensor *, int> wlevel, rlevel; // per storage root: latest writer / reader level
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = node(i);
        int lvl = 0;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = t->src[s];
            if (!src) {
                continue;
            }
            auto it = idx.find(src);
            if (it != idx.end()) {
                lvl = std::max(lvl, level[it->second] + 1);
            }
            auto w = wlevel.find(root(src));
            if (w != wlevel.end()) {
                lvl = std::max(lvl, w->second + 1);
            }
        }
        if (writes(t)) {
            const ggml_tensor * r = root(t);
            if (wlevel.count(r)) { lvl = std::max(lvl, wlevel[r] + 1); }
            if (rlevel.count(r)) { lvl = std::max(lvl, rlevel[r] + 1); }
            wlevel[r] = std::max(wlevel.count(r) ? wlevel[r] : 0, lvl);
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (t->src[s]) {
                const ggml_tensor * r = root(t->src[s]);
                rlevel[r] = std::max(rlevel.count(r) ? rlevel[r] : 0, lvl);
            }
        }
        level[i] = lvl;
    }
    auto peak = [&](const std::vector<int> & order) {
        std::vector<int> pos(n);
        for (int k = 0; k < n; ++k) {
            pos[order[k]] = k;
        }
        // a node that owns memory (not a view, not a leaf) lives from its step to its last reader's step (through
        // views); graph outputs to the end
        std::unordered_map<const ggml_tensor *, int> last;
        for (int i = 0; i < n; ++i) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (node(i)->src[s]) {
                    const ggml_tensor * r = root(node(i)->src[s]);
                    if (idx.count(r)) {
                        last[r] = std::max(last.count(r) ? last[r] : 0, pos[i]);
                    }
                }
            }
            if (node(i)->flags & GGML_TENSOR_FLAG_OUTPUT) {
                last[root(node(i))] = n;
            }
        }
        std::vector<int64_t> delta(n + 2, 0);
        for (int i = 0; i < n; ++i) {
            const ggml_tensor * t = node(i);
            if (t->view_src || t->op == GGML_OP_NONE) {
                continue;
            }
            const int end = last.count(t) ? std::max(last[t], pos[i]) : pos[i];
            delta[pos[i]] += ggml_nbytes(t);
            delta[end + 1] -= ggml_nbytes(t);
        }
        int64_t cur = 0, best = 0;
        for (int k = 0; k <= n; ++k) {
            cur += delta[k];
            best = std::max(best, cur);
        }
        return size_t(best);
    };
    // graph inputs and their views as read by compute nodes: a non-CPU backend's scheduler copies each distinct one
    // into the split's compute buffer, alive for the whole split (848c757ad: one mask view per head and layer, 2.7 GiB
    // at 9K cells on Vulkan); view nodes themselves only pass the input through
    auto passes = [](const ggml_tensor * t) {
        return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
    };
    std::set<const ggml_tensor *> copies;
    input_views = 0;
    for (int i = 0; i < n; ++i) {
        if (passes(node(i))) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node(i)->src[s];
            if (src && (root(src)->flags & GGML_TENSOR_FLAG_INPUT) && copies.insert(src).second) {
                input_views += ggml_nbytes(src);
            }
        }
    }

    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) {
        order[i] = i;
    }
    in_order = peak(order);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return level[a] < level[b]; });
    any_order = peak(order);
}

int64_t llama_ridgefill_extra_nodes(const llama_ridgefill * rf, const llama_hparams & hparams, int64_t n_ctx, int64_t n_ubatch) {
    if (!rf || n_ctx <= 0 || n_ubatch <= 0) {
        return 0;
    }
    int64_t res = 0;
    for (uint32_t il = std::max(rf->split, 0); il < hparams.n_layer(); ++il) {
        if (hparams.is_recr(il) || hparams.n_head_kv(il) == 0) {
            continue;
        }
        // S1 probe: per query head and query block a score block (views, follow, mul_mat, softmax, PV, out_prod,
        // adds), the per-head output merge; vtail and masks
        const int64_t qb = llama_ridgefill_query_block(n_ctx, n_ubatch, LLAMA_RIDGEFILL_PROBE_BLOCK);
        const int64_t blocks = int64_t(hparams.n_head(il))*((n_ubatch + qb - 1)/qb);
        res += 16*blocks + 8*hparams.n_head_kv(il) + 128;
    }
    return res + 256;
}

ggml_tensor * build_ridgefill_s1_attn(llm_graph_context & g, const llama_model & model, const llm_graph_input_ridgefill_part * inp,
        ggml_tensor * cur, ggml_tensor * inp_pos, llm_graph_input_attn_kv * inp_attn, int il, ggml_tensor ** mass) {
    // qwen35's build_layer_attn for the window rows, with the attention written out (non-FA math, as 22ef3df4a) so the
    // softmax weights give the mass: visible cells = the probe mask (causal, the unrun tail rows hidden)
    const auto & hparams = g.hparams;
    const auto & layer   = model.layers[il];
    ggml_context * ctx0  = g.ctx0;
    const int64_t n_tokens    = g.n_tokens;
    const int64_t n_embd_head = hparams.n_embd_head_v();

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto [Qfull, Kcur, Vcur] = g.build_qkv(layer, cur,
            n_embd_head * 2, g.n_head,
            n_embd_head,     g.n_head_kv,
            n_embd_head,     g.n_head_kv,
            il, false);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qfull, n_embd_head, g.n_head, n_tokens,
            ggml_element_size(Qfull) * n_embd_head * 2,
            ggml_element_size(Qfull) * n_embd_head * 2 * g.n_head, 0);
    Qcur = g.build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, g.n_head_kv, n_tokens);
    Kcur = g.build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qfull, n_embd_head, g.n_head, n_tokens,
            ggml_element_size(Qfull) * n_embd_head * 2,
            ggml_element_size(Qfull) * n_embd_head * 2 * g.n_head,
            ggml_element_size(Qfull) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * g.n_head, n_tokens);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, g.n_head_kv, n_tokens);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr, g.n_rot, sections, g.rope_type, g.n_ctx_orig, g.freq_base,
            g.freq_scale, g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr, g.n_rot, sections, g.rope_type, g.n_ctx_orig, g.freq_base,
            g.freq_scale, g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // as build_attn: the cache's rotations, then store the window's K/V (the later window ubatches read them)
    if (inp_attn->self_k_rot) {
        Qcur = llama_mul_mat_hadamard(ctx0, Qcur, inp_attn->self_k_rot);
        Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
    }
    if (inp_attn->self_v_rot) {
        Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
    }
    ggml_build_forward_expand(g.gf, Qcur);
    ggml_build_forward_expand(g.gf, Vcur);
    ggml_build_forward_expand(g.gf, Kcur);

    const auto * mctx_cur = inp_attn->mctx;
    ggml_build_forward_expand(g.gf, mctx_cur->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
    ggml_build_forward_expand(g.gf, mctx_cur->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    if (g.ridgefill_params.certificate) {
        const auto * rf = g.ridgefill;
        ggml_tensor * error = nullptr;
        if (!rf->streamed) {
            ggml_tensor * x = build_ridgefill_h_input(g, inp);
            if (rf->w[il]) { x = ggml_add(ctx0, ggml_mul_mat(ctx0, rf->w[il], x), rf->b[il]); }
            x = g.build_norm(x, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            auto predicted = g.build_qkv(layer, x, n_embd_head*2, g.n_head, n_embd_head, g.n_head_kv,
                    n_embd_head, g.n_head_kv, il, false);
            ggml_tensor * kp = ggml_reshape_3d(ctx0, predicted.k, n_embd_head, g.n_head_kv, n_tokens);
            kp = g.build_norm(kp, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
            kp = ggml_rope_multi(ctx0, kp, inp_pos, nullptr, g.n_rot, sections, g.rope_type, g.n_ctx_orig, g.freq_base,
                    g.freq_scale, g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
            if (inp_attn->self_k_rot) { kp = llama_mul_mat_hadamard(ctx0, kp, inp_attn->self_k_rot); }
            auto stored = [&](ggml_tensor * t) { return ggml_cast(ctx0, ggml_cast(ctx0, t, k->type), GGML_TYPE_F32); };
            // Read the written cache; an extra Kcur consumer would break the existing fused store.
            ggml_tensor * exact = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, k, n_embd_head*g.n_head_kv, k->ne[2]), inp->cert_rows);
            exact = ggml_reshape_3d(ctx0, exact, n_embd_head, g.n_head_kv, n_tokens);
            error = ggml_sub(ctx0, stored(kp), exact);
            error = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, error)));
        }
        ggml_tensor * queries = ggml_cont(ctx0, Qcur);
        ggml_set_output(queries);
        if (error) { ggml_set_output(error); }
        ggml_build_forward_expand(g.gf, queries);
        if (error) { ggml_build_forward_expand(g.gf, error); }
        g.res->t_rf_calibration.push_back({il, queries, error});
    }

    ggml_tensor * v = mctx_cur->get_v(ctx0, il);
    GGML_ASSERT(k->ne[3] == 1 && "RidgeFill S1 probe: one KV stream");

    const bool v_trans = v->nb[1] > v->nb[2];

    // per query head (all of the ubatch's queries while a head's scores fit LLAMA_RIDGEFILL_PROBE_BLOCK, else equal
    // query blocks): the matrices of the unsliced attention, one head at a time, so the per-element arithmetic is the
    // unsliced one on every backend (the tile and split-k choices depend on m, n, k only); each block follows the
    // previous one (llama_ridgefill_follow), so one block of scores is alive at a time
    const int64_t d = n_embd_head, n_head = g.n_head, n_head_kv = g.n_head_kv, rep = n_head/n_head_kv;
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [d, n_kv, n_head_kv]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    const int64_t n_cells = k->ne[1];
    const int64_t n_kv = inp->s1_rows ? inp->s1_rows->ne[0] : n_cells;
    GGML_ASSERT(inp->s1_mask->ne[0] == n_kv && inp->s1_mask->ne[1] == n_tokens);
    const int64_t qb = inp->s1_qb; // the query blocks of the mask / ones views made once per graph

    ggml_tensor * dep = nullptr;
    ggml_tensor * m_layer = nullptr;
    std::vector<ggml_tensor *> heads; // [d, n_tokens, 1] per query head
    for (int64_t kh = 0; kh < n_head_kv; ++kh) {
        ggml_tensor * kg = ggml_view_3d(ctx0, k, d, n_cells, 1, k->nb[1], k->nb[2], kh*k->nb[2]);
        ggml_tensor * vg = ggml_view_3d(ctx0, v, v->ne[0], v->ne[1], 1, v->nb[1], v->nb[2], kh*v->nb[2]);
        if (inp->s1_rows) {
            // Exclude other sequences' cells before attention, keeping this prompt in position order.
            kg = ggml_cast(ctx0, ggml_get_rows(ctx0, kg, inp->s1_rows), k->type);
            if (v_trans) { vg = ggml_cont(ctx0, ggml_transpose(ctx0, vg)); }
            vg = ggml_cast(ctx0, ggml_get_rows(ctx0, vg, inp->s1_rows), v->type);
            vg = ggml_cont(ctx0, ggml_transpose(ctx0, vg));
        } else if (!v_trans) {
            // [n_kv, d, 1], as the unsliced cont(transpose(v)) (not chained: at most one copy per key head is alive)
            vg = ggml_cont(ctx0, ggml_transpose(ctx0, vg));
        }
        for (int64_t h = kh*rep; h < (kh + 1)*rep; ++h) {
            ggml_tensor * out_h = nullptr;
            for (int64_t q0 = 0; q0 < n_tokens; q0 += qb) {
                const int64_t nq = std::min(qb, n_tokens - q0);
                ggml_tensor * qp = ggml_permute(ctx0, llama_ridgefill_follow(ctx0, Qcur, dep), 0, 2, 1, 3); // [d, n_tokens, n_head]
                ggml_tensor * qv = ggml_view_3d(ctx0, qp, d, nq, 1, qp->nb[1], qp->nb[2], q0*qp->nb[1] + h*qp->nb[2]);

                ggml_tensor * kq = ggml_mul_mat(ctx0, kg, qv); // [n_kv, nq, 1]
                ggml_prec_set_acc(kq, GGML_PREC_F32);
                ggml_tensor * probs = ggml_soft_max_ext(ctx0, kq, inp->s1_mask_blk[q0/qb], kq_scale, 0.0f);
                g.cb(probs, "ridgefill_s1_probs", il);

                ggml_tensor * o = ggml_mul_mat(ctx0, vg, probs); // [d, nq, 1]
                ggml_tensor * m = ggml_out_prod(ctx0, ggml_reshape_2d(ctx0, probs, n_kv, nq), inp->s1_ones_blk[q0/qb]); // [n_kv, 1]
                m_layer = m_layer ? ggml_add(ctx0, m_layer, m) : m;
                out_h = out_h ? ggml_concat(ctx0, out_h, o, 1) : o;
                // the next block follows both consumers of these scores
                dep = ggml_add(ctx0, ggml_view_1d(ctx0, o, 1, 0), ggml_view_1d(ctx0, m, 1, 0));
            }
            heads.push_back(out_h);
        }
    }
    // [d, n_tokens, n_head] from the heads (a balanced merge: copies only), then as the unsliced kqv
    while (heads.size() > 1) {
        std::vector<ggml_tensor *> next;
        for (size_t i = 0; i + 1 < heads.size(); i += 2) {
            next.push_back(ggml_concat(ctx0, heads[i], heads[i + 1], 2));
        }
        if (heads.size() % 2) {
            next.push_back(heads.back());
        }
        heads.swap(next);
    }
    ggml_tensor * kqv = heads[0];
    ggml_tensor * out = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    out = ggml_cont_2d(ctx0, out, out->ne[0]*out->ne[1], out->ne[2]*out->ne[3]);

    if (inp_attn->self_v_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, inp_attn->self_v_rot);
    }

    out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, gate));
    out = g.build_lora_mm(layer.wo, out, layer.wo_s);

    // the layer's mass per cell, summed over the queries and heads; expanded now, so it is computed with this layer
    g.cb(m_layer, "ridgefill_s1_mass", il);
    ggml_build_forward_expand(g.gf, m_layer);
    *mass = *mass ? ggml_add(ctx0, *mass, m_layer) : m_layer;

    return out;
}

void build_ridgefill_vtail(llm_graph_context & g, const llama_model & model, const llm_graph_input_ridgefill_part * inp,
        ggml_tensor * cur, ggml_tensor * h_S, llm_graph_input_attn_kv * inp_attn, int il, ggml_tensor ** scales) {
    const auto & hparams = g.hparams;
    const auto & layer   = model.layers[il];
    const llama_ridgefill * rf = g.ridgefill;
    ggml_context * ctx0  = g.ctx0;
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_head_kv   = g.n_head_kv;
    const int64_t c           = inp->params.vtail_rows;
    GGML_ASSERT(c >= 1 && c <= g.n_tokens && rf->w[il]);

    // V of the first c rows: from their exact input (cur = attn_norm of it) and from the projector on their h_S
    auto v_of = [&](ggml_tensor * x) {
        auto [Q, K, V] = g.build_qkv(layer, x, n_embd_head * 2, g.n_head, n_embd_head, n_head_kv, n_embd_head, n_head_kv, il, false);
        GGML_UNUSED(Q);
        GGML_UNUSED(K);
        return ggml_reshape_3d(ctx0, ggml_cont(ctx0, V), n_embd_head, n_head_kv, c);
    };
    ggml_tensor * x_true = ggml_view_2d(ctx0, cur, cur->ne[0], c, cur->nb[1], 0);
    ggml_tensor * x_fill = ggml_view_2d(ctx0, h_S, h_S->ne[0], c, h_S->nb[1], 0);
    auto * weight = g.ridgefill_params.stream_layer >= 0 ? g.ridgefill_params.stream_w : rf->w[il];
    auto * bias = g.ridgefill_params.stream_layer >= 0 ? g.ridgefill_params.stream_b : rf->b[il];
    x_fill = ggml_add(ctx0, ggml_mul_mat(ctx0, weight, x_fill), bias);
    x_fill = g.build_norm(x_fill, layer.attn_norm, nullptr, LLM_NORM_RMS, il);

    auto norm_of = [&](ggml_tensor * v) {                     // [1, n_head_kv, c]
        return ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, v)));
    };
    ggml_tensor * t = norm_of(v_of(x_true));
    ggml_tensor * f = ggml_clamp(ctx0, norm_of(v_of(x_fill)), 1e-30f, INFINITY);
    ggml_tensor * r = ggml_reshape_2d(ctx0, ggml_div(ctx0, t, f), n_head_kv, c);
    ggml_tensor * s = ggml_mean(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, r)));        // [1, n_head_kv]
    s = ggml_reshape_1d(ctx0, s, n_head_kv);
    g.cb(s, "ridgefill_vtail_scale", il);

    // the bulk prefix cells' V times s_h, before this layer's tail attention reads the cache
    ggml_tensor * v = inp_attn->mctx->get_v(g.ctx0, il);
    GGML_ASSERT(v->ne[3] == 1 && "RidgeFill vtail: one KV stream");
    const bool v_trans = v->nb[1] > v->nb[2];
    const int64_t n_kv = inp->vt_bulk->ne[0];

    ggml_tensor * mult;
    if (!v_trans) {                                           // v [d, n_head_kv, n_kv]
        ggml_tensor * b = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inp->vt_bulk, 1, 1, n_kv), 1, n_head_kv, n_kv, 1);
        mult = ggml_add(ctx0, ggml_mul(ctx0, b, ggml_reshape_3d(ctx0, s, 1, n_head_kv, 1)), ggml_reshape_3d(ctx0, inp->vt_keep, 1, 1, n_kv));
    } else {                                                  // v [n_kv, n_head_kv, d]
        ggml_tensor * b = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inp->vt_bulk, n_kv, 1, 1), n_kv, n_head_kv, 1, 1);
        mult = ggml_add(ctx0, ggml_mul(ctx0, b, ggml_reshape_3d(ctx0, s, 1, n_head_kv, 1)), ggml_reshape_3d(ctx0, inp->vt_keep, n_kv, 1, 1));
    }
    // cast, scale and store back a bounded run of cells at a time: the arithmetic of the unsliced cast / mul / cpy
    // (an in-place F16 mul rounds differently on Vulkan), without an n_kv-sized temporary. The runs write disjoint views
    // of one cache tensor, which a reordering backend keeps in order.
    const int64_t per_cell = n_head_kv*v->ne[v_trans ? 2 : 0]; // elements of one cell
    const int64_t step = std::max<int64_t>(1, (int64_t(1) << 22)/per_cell);
    for (int64_t c0 = 0; c0 < n_kv; c0 += step) {
        const int64_t nc = std::min(step, n_kv - c0);
        ggml_tensor * vc;
        ggml_tensor * mc;
        if (!v_trans) { // v [d, n_head_kv, n_kv]
            vc = ggml_view_3d(ctx0, v, v->ne[0], v->ne[1], nc, v->nb[1], v->nb[2], c0*v->nb[2]);
            mc = ggml_view_3d(ctx0, mult, 1, n_head_kv, nc, mult->nb[1], mult->nb[2], c0*mult->nb[2]);
        } else {        // v [n_kv, n_head_kv, d]
            vc = ggml_view_3d(ctx0, v, nc, v->ne[1], v->ne[2], v->nb[1], v->nb[2], c0*v->nb[0]);
            mc = ggml_view_3d(ctx0, mult, nc, n_head_kv, 1, mult->nb[1], mult->nb[2], c0*mult->nb[0]);
        }
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx0, ggml_mul(ctx0, ggml_cast(ctx0, vc, GGML_TYPE_F32), mc), vc));
    }

    ggml_tensor * col = ggml_reshape_2d(ctx0, s, n_head_kv, 1);
    *scales = *scales ? ggml_concat(ctx0, *scales, col, 1) : col;
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
