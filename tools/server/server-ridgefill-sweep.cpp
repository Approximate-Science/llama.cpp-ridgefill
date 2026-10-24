#include "server-ridgefill-sweep.h"

#include "../../src/llama-ridgefill-sweep.h"
#include "../../src/llama-context.h"
#include "../../src/llama-memory-hybrid.h"
#include "../../src/llama-model.h"
#include "../../src/llama-ridgefill.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cinttypes>
#include <cstring>
#include <iterator>
#include <stdexcept>

server_ridgefill_sweep::server_ridgefill_sweep() = default;
server_ridgefill_sweep::~server_ridgefill_sweep() = default;

const char * server_ridgefill_sweep::phase_name() const {
    static const char * names[] = {"lower", "mass", "rows", "s1_fill", "s1_probe", "s1_rows", "prefix", "stream_tail", "tail", "external", "done"};
    return names[int(next)];
}

int32_t server_ridgefill_sweep::next_tokens(llama_context * ctx, const server_ridgefill_sweep_params & params) const {
    int32_t end;
    switch (next) {
        case phase::lower:    end = params.n_tokens; break;
        case phase::s1_fill:
        case phase::prefix:   end = params.n_prefix; break;
        case phase::s1_probe: end = win_end; break;
        case phase::stream_tail: return stream_calib_rows;
        case phase::tail:     end = params.n_tokens - int(params.defer_last_token); break;
        default: return 0;
    }
    int32_t count = std::min({params.n_batch, int32_t(llama_n_ubatch(ctx)), end - position});
    if (next == phase::tail && params.vtail && position == params.n_prefix) {
        count = std::max(count, std::min(params.vtail_rows, params.n_tokens - params.n_prefix));
    }
    return count;
}

int32_t server_ridgefill_sweep_api_version() {
#if defined(LLAMA_RIDGEFILL_API_VERSION) && LLAMA_RIDGEFILL_API_VERSION >= 2
    return llama_ridgefill_api_version();
#else
    return 1;
#endif
}

void server_ridgefill_sweep::prepare_cost(const llama_model & model, const server_ridgefill_sweep_params & params) {
    const auto & hp = model.hparams;
    const double h = hp.n_embd, p = params.n_prefix, w = params.s1 ? params.s1_window : 0;
    cost.s1_fill_rows = params.s1 ? params.n_prefix : 0;
    cost.s1_probe_rows = w;
    cost.vtail_rows = params.vtail ? std::min(params.vtail_rows, params.n_tokens - params.n_prefix) : 0;
    int32_t heads = 0;
    for (uint32_t il = params.split; il < hp.n_layer(); ++il) {
        const double mlp = 3 * h * hp.n_ff(il);
        if (hp.is_recr(il)) {
            if (!hp.ssm_dt_rank || hp.ssm_d_inner % hp.ssm_dt_rank) {
                throw std::runtime_error("RidgeFill S1: invalid GDN value heads for cost reporting");
            }
            const double nk = hp.ssm_n_group, nv = hp.ssm_dt_rank, dk = hp.ssm_d_state, dv = hp.ssm_d_inner / hp.ssm_dt_rank;
            const double conv = 2 * nk * dk + nv * dv;
            const double linear = h * conv + 2 * h * nv + conv * hp.ssm_d_conv + 2 * h * nv * dv + 4 * nv * dk * dv;
            cost.s1_probe_macs += w * (linear + mlp);
        } else {
            const double hq = hp.n_head(il), hkv = hp.n_head_kv(il), d = hp.n_embd_head_k(il);
            if (d != hp.n_embd_head_v(il)) {
                throw std::runtime_error("RidgeFill S1/vtail: cost reporting requires equal K/V head widths");
            }
            heads += hp.n_head_kv(il);
            cost.s1_fill_macs += params.s1 ? p * (h * h + 2 * h * hkv * d) : 0;
            cost.s1_probe_macs += w * (3 * h * hq * d + 2 * h * hkv * d + mlp);
            cost.vtail_macs += cost.vtail_rows * (h * h + 2 * h * hkv * d);
        }
    }
    if ((params.s1 || params.vtail) && heads == 0) {
        throw std::runtime_error("RidgeFill S1/vtail: no late attention layers");
    }
    if (params.vtail) { vtail_scales.resize(heads); }
}

void server_ridgefill_sweep::decode(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        int32_t count, bool output_all, bool output_last, int32_t logits_begin, int32_t logits_end) {
    if (tail_hidden && position >= hidden_keep_end) {
        ctx->ridgefill_seq_stage_tail(seq, position, count);
        if (!ggml_backend_buffer_is_host(tail_hidden->tensor()->buffer)) { hidden_upload_bytes += size_t(count)*hidden->tensor()->nb[1]; }
    }
    llama_batch batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    n_outputs = 0;
    output_position = -1;
    for (int32_t i = 0; i < count; ++i) {
        batch.token[i] = tokens[position + i];
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = output_all || (output_last && i == count - 1) ||
                (logits_begin >= 0 && position + i >= logits_begin && position + i < logits_end);
        if (batch.logits[i]) {
            if (n_outputs == 0) { output_position = position + i; }
            ++n_outputs;
        }
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("RidgeFill quality+qattn: sweep decode failed with code " + std::to_string(rc));
    }
    position += count;
    n_step_tokens = count;
}

void server_ridgefill_sweep::sweep_lower(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        const server_ridgefill_sweep_params & params) {
    const auto & model = ctx->get_model();
    if (!hidden) {
        if (params.split <= 0 || params.split >= (int32_t) model.hparams.n_layer() || params.n_embd != (int32_t) model.hparams.n_embd) {
            throw std::runtime_error("RidgeFill quality+qattn: invalid split or embedding width");
        }
        if (params.s1 || params.vtail) { prepare_cost(model, params); }
        stream_stride = params.n_tokens;
        stream_split = params.split;
        if (ctx->get_ridgefill_stream() && params.n_tokens > INT32_MAX/int32_t(model.hparams.n_layer() - params.split)) {
            throw std::runtime_error("RidgeFill stream: cursor exceeds the integer range");
        }
        win_end = params.qattn_end >= 0 ? params.qattn_end : (params.qattn_blind ? params.n_prefix : params.n_tokens);
        win_begin = std::max(0, win_end - params.qattn_window);
        if (params.qattn_share > 0 && params.qattn_window_mode == "turn") {
            const std::string header = "<|im_start|>user\n";
            const auto * vocab = llama_model_get_vocab(&model);
            const int32_t n = -llama_tokenize(vocab, header.data(), header.size(), nullptr, 0, false, true);
            if (n <= 0 || n > 64) {
                throw std::runtime_error("RidgeFill qattn-window-mode turn: cannot tokenize the user header");
            }
            std::vector<llama_token> marker(n);
            if (llama_tokenize(vocab, header.data(), header.size(), marker.data(), n, false, true) != n) {
                throw std::runtime_error("RidgeFill qattn-window-mode turn: cannot tokenize the user header");
            }
            const int32_t start = llama_ridgefill_last_marker(tokens, win_end, marker.data(), n);
            if (start < 0) {
                throw std::runtime_error("RidgeFill qattn-window-mode turn: no user header before the window endpoint");
            }
            win_begin = std::max(start + n, win_end - params.qattn_turn_cap);
            if (win_begin >= win_end) {
                throw std::runtime_error("RidgeFill qattn-window-mode turn: empty user turn before the window endpoint");
            }
        }
        // the question-attention layers: the last 4 attention layers below the split (the reference rule: kva
        // q27_quality.py, lane T's rf27 references, the projector manifest's qattn_rule); {3, 7, 11, 15} at split 16
        std::vector<int32_t> qattn_layers;
        for (int32_t il = 0; il < params.split && params.qattn_share > 0; ++il) {
            if (!model.hparams.is_recr(il)) {
                qattn_layers.push_back(il);
            }
        }
        if (qattn_layers.size() > 4) {
            qattn_layers.erase(qattn_layers.begin(), qattn_layers.end() - 4);
        }
        for (int32_t il : qattn_layers) {
            layers.push_back(il);
            windows.push_back(std::make_unique<llama_ridgefill_sweep_buffer>(model.select_buft(il),
                    params.n_embd, win_end - win_begin, GGML_TYPE_F32));
        }
        if (layers.empty() && params.qattn_share > 0) {
            throw std::runtime_error("RidgeFill quality+qattn: split has no lower attention layers");
        }
        hidden = std::make_unique<llama_ridgefill_sweep_buffer>(model.select_buft(params.split),
                params.n_embd, params.n_tokens, params.type, params.max_bytes);
        std::vector<ggml_tensor *> buffers;
        for (const auto & window : windows) { buffers.push_back(window->tensor()); }
        auto part = llama_ridgefill_part_default_params();
        part.part = LLAMA_RIDGEFILL_PART_LOWER;
        part.h_buf = hidden->tensor();
        part.n_win_layers = layers.size();
        part.win_layers = layers.data();
        part.win_bufs = buffers.data();
        part.win_begin = layers.empty() ? 0 : win_begin;
        part.win_n = layers.empty() ? 0 : win_end - win_begin;
        if (llama_ridgefill_seq_set_part(ctx, seq, &part) < 0) {
            throw std::runtime_error("RidgeFill quality+qattn: LOWER part was refused");
        }
        active = true;
    }
    const int32_t count = std::min({params.n_batch, int32_t(llama_n_ubatch(ctx)), params.n_tokens - position});
    decode(ctx, seq, tokens, count, false, false);
    n_lower = position;
}

void server_ridgefill_sweep::question_mass(llama_context * ctx, llama_seq_id seq, const server_ridgefill_sweep_params & params) {
    if (ctx->get_ridgefill_stream()) {
        ctx->synchronize();
        const bool profiling = params.projector && params.projector->profiling;
        const int64_t begin = profiling ? ggml_time_us() : 0;
        for (auto & window : windows) {
            if (ggml_backend_buffer_is_host(window->tensor()->buffer)) { continue; }
            auto host = std::make_unique<llama_ridgefill_sweep_buffer>(ggml_backend_cpu_buffer_type(),
                    params.n_embd, win_end - win_begin, GGML_TYPE_F32);
            ggml_backend_tensor_get(window->tensor(), host->tensor()->data, 0, window->size_bytes());
            window = std::move(host);
        }
        if (profiling) { llama_ridgefill_profile(params.projector, "window_offload", -1, 0, begin, ggml_time_us()); }
    }
    std::vector<ggml_tensor *> buffers;
    for (const auto & window : windows) { buffers.push_back(window->tensor()); }
    mass = llama_ridgefill_sweep_question_mass(*ctx, seq, params.n_tokens, win_end, layers, buffers);
}

void server_ridgefill_sweep::select_rows(llama_context * ctx, const llama_token * tokens,
        const std::vector<llama_pos> & base_rows, const server_ridgefill_sweep_params & params) {
    if (params.qattn_share > 0 && mass.size() != size_t(params.n_tokens)) {
        throw std::runtime_error("RidgeFill quality+qattn: question mass does not cover the prompt");
    }
    if (!std::is_sorted(base_rows.begin(), base_rows.end()) ||
            std::adjacent_find(base_rows.begin(), base_rows.end()) != base_rows.end() ||
            (!base_rows.empty() && (base_rows.front() < 0 || base_rows.back() >= params.n_prefix))) {
        throw std::runtime_error("RidgeFill quality+qattn: base exact rows must be sorted unique prefix positions");
    }
    const auto question_rows = params.qattn_share > 0 ? llama_ridgefill_sweep_top_rows(mass, params.n_prefix, params.qattn_share) : std::vector<int32_t>();
    rows.clear();
    std::set_union(base_rows.begin(), base_rows.end(), question_rows.begin(), question_rows.end(), std::back_inserter(rows));
    n_base = rows.size();
    if (params.lexical) {
        lexical = common_ridgefill_lexical_rows(llama_model_get_vocab(llama_get_model(ctx)), tokens,
                params.n_tokens, params.n_prefix, params.qattn_end, params.lexical_options);
        if (!params.s1) {
            std::vector<llama_pos> forced;
            std::set_union(base_rows.begin(), base_rows.end(), lexical.rows.begin(), lexical.rows.end(), std::back_inserter(forced));
            rows = llama_ridgefill_sweep_top_k(mass, params.n_prefix, n_base, forced);
        }
    }
}

#if defined(LLAMA_RIDGEFILL_API_VERSION) && LLAMA_RIDGEFILL_API_VERSION >= 2
static llama_kv_cache & sweep_cache(llama_context * ctx) {
    auto * memory = dynamic_cast<llama_memory_hybrid *>(ctx->get_memory());
    if (!memory) { throw std::runtime_error("RidgeFill S1/vtail: cost reporting requires a hybrid cache"); }
    return *memory->get_mem_attn();
}

static uint32_t sweep_cache_rows(const llama_kv_cache & cache, llama_seq_id seq) {
    llama_kv_cache::slot_info slot = {};
    slot.strm.push_back(cache.get_stream(seq));
    return cache.get_n_kv(slot);
}

void server_ridgefill_sweep::sweep_s1_fill(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        const server_ridgefill_sweep_params & params) {
    auto * maps = ctx->get_ridgefill_stream();
    const bool first = position == 0 && stream_layer < 0;
    if (first) {
        auto part = llama_ridgefill_part_default_params();
        part.part = LLAMA_RIDGEFILL_PART_S1_FILL;
        part.h_buf = hidden->tensor();
        if (llama_ridgefill_seq_set_part(ctx, seq, &part) < 0) {
            throw std::runtime_error("RidgeFill S1: S1_FILL was refused");
        }
        if (maps) {
            stream_layer = params.split;
            while (ctx->get_model().hparams.is_recr(stream_layer)) { ++stream_layer; }
            ctx->ridgefill_seq_stream_layer(seq, stream_layer, nullptr, params.n_prefix);
        }
    }
    if (maps) {
        int32_t next_layer = stream_layer + 1;
        while (next_layer < int32_t(ctx->get_model().hparams.n_layer()) && ctx->get_model().hparams.is_recr(next_layer)) { ++next_layer; }
        maps->stage(*ctx, stream_layer, next_layer < int32_t(ctx->get_model().hparams.n_layer()) ? next_layer : -1);
    }
    decode(ctx, seq, tokens, std::min({params.n_batch, int32_t(llama_n_ubatch(ctx)), params.n_prefix - position}), false, false);
    // S1 only reads hS; LOWER is complete and the first fill graph is submitted.
    if (first && maps) { prefetch_hidden(ctx, params); }
    if (maps && position == params.n_prefix) { advance_stream(ctx, seq, params, true); }
}

void server_ridgefill_sweep::sweep_s1_probe(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        const server_ridgefill_sweep_params & params) {
    if (position == win_end - params.s1_window) {
        auto part = llama_ridgefill_part_default_params();
        part.part = LLAMA_RIDGEFILL_PART_S1_PROBE;
        part.h_buf = hidden->tensor();
        part.s1_begin = position;
        part.s1_end = win_end;
        if (llama_ridgefill_seq_set_part(ctx, seq, &part) < 0) {
            throw std::runtime_error("RidgeFill S1: S1_PROBE was refused");
        }

    }
    const int32_t count = std::min({params.n_batch, int32_t(llama_n_ubatch(ctx)), win_end - position});
    const auto & cp = ctx->get_cparams();
    const uint32_t keys = cp.kv_unified && cp.n_seq_max > 1 ? uint32_t(params.n_tokens) : sweep_cache_rows(sweep_cache(ctx), seq);
    const uint64_t pairs = uint64_t(count) * keys;
    decode(ctx, seq, tokens, count, false, false);
    cost.s1_probe_kv_pairs += pairs;
    const auto & hp = ctx->get_model().hparams;
    for (uint32_t il = params.split; il < hp.n_layer(); ++il) {
        if (!hp.is_recr(il)) { cost.s1_probe_macs += 2.0 * hp.n_head(il) * hp.n_embd_head_k(il) * pairs; }
    }
}

void server_ridgefill_sweep::select_s1_rows(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        const server_ridgefill_sweep_params & params) {
    s1_mass.resize(params.n_prefix);
    if (llama_ridgefill_seq_get_s1_mass(ctx, seq, s1_mass.data(), s1_mass.size()) != params.n_prefix) {
        throw std::runtime_error("RidgeFill S1: complete prefix mass is unavailable");
    }
    const double budget = std::min(double(params.n_prefix), params.s1_factor * n_base);
    int32_t count = int32_t(std::floor(budget));
    const double fraction = budget - count;
    if (fraction > 0.5 || (fraction == 0.5 && (count & 1))) { ++count; }
    std::vector<llama_pos> special;
    if (params.special_exact) {
        auto select = llama_ridgefill_select_default_params(params.projector);
        select.share = 0; select.digits = false; select.special_exact = true; select.header_rows = 0;
        special.resize(params.n_prefix);
        const int32_t n = llama_ridgefill_select_rows_ext(params.projector, tokens, params.n_prefix, &select, special.data());
        if (n < 0 || n > params.n_prefix) { throw std::runtime_error("RidgeFill S1: special-row selection was refused"); }
        special.resize(n);
    }
    const auto selected = llama_ridgefill_sweep_top_k(s1_mass, params.n_prefix, std::max(1, count), lexical.rows);
    rows.clear();
    std::set_union(selected.begin(), selected.end(), special.begin(), special.end(), std::back_inserter(rows));

}


#endif

int32_t server_ridgefill_sweep::next_work(llama_context * ctx, llama_seq_id seq, const server_ridgefill_sweep_params & params, int32_t capacity) const {
    if (next == phase::external || next == phase::done) { return 0; }
    const bool profiling = params.projector && params.projector->profiling;
    const int64_t begin = profiling ? ggml_time_us() : 0;
    const auto & model = ctx->get_model();
    const auto & hp = model.hparams;
    const double h = hp.n_embd;
    double keys = params.n_tokens;
#if defined(LLAMA_RIDGEFILL_API_VERSION) && LLAMA_RIDGEFILL_API_VERSION >= 2
    keys = std::max(keys, double(sweep_cache_rows(sweep_cache(ctx), seq)));
#endif
    const int32_t count = next_tokens(ctx, params);
    const bool layerwise = ctx->get_ridgefill_stream() && (next == phase::s1_fill || next == phase::prefix || next == phase::stream_tail);
    int32_t layer_index = stream_layer >= 0 ? stream_layer : params.split;
    if (layerwise && next == phase::s1_fill) {
        while (layer_index < int32_t(hp.n_layer()) && hp.is_recr(layer_index)) { ++layer_index; }
    }
    int32_t exact = 0;
    if (next == phase::prefix) {
        exact = std::lower_bound(rows.begin(), rows.end(), position + count) - std::lower_bound(rows.begin(), rows.end(), position);
    }
    double stock = 0, lower = 0, late = 0, fill = 0, extra = 0, attn_fill = 0;
    double stock_bytes = 0, lower_bytes = 0, late_bytes = 0, fill_bytes = 0, extra_bytes = 0, attn_bytes = 0;
    auto bytes = [](std::initializer_list<const ggml_tensor *> tensors) {
        double total = 0;
        for (const auto * tensor : tensors) { if (tensor) { total += ggml_nbytes(tensor); } }
        return total;
    };
    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
        const double mlp = 3 * h * hp.n_ff(il);
        double linear, bulk, attention = 0;
        if (hp.is_recr(il)) {
            if (!hp.ssm_dt_rank || hp.ssm_d_inner % hp.ssm_dt_rank) { throw std::runtime_error("RidgeFill budget: invalid GDN heads"); }
            const double nk = hp.ssm_n_group, nv = hp.ssm_dt_rank, dk = hp.ssm_d_state, dv = hp.ssm_d_inner / hp.ssm_dt_rank;
            const double conv = 2 * nk * dk + nv * dv;
            const double inproj = h * conv + 2 * h * nv + conv * hp.ssm_d_conv;
            linear = inproj + 2 * h * nv * dv + 4 * nv * dk * dv;
            bulk = inproj + 3 * nv * dk * dv;
        } else {
            const double hq = hp.n_head(il), hkv = hp.n_head_kv(il), d = hp.n_embd_head_k(il);
            bulk = 2 * h * hkv * d;
            linear = 3 * h * hq * d + bulk;
            attention = 2 * hq * d * keys;
            if (il >= uint32_t(params.split) && (!layerwise || il == uint32_t(layer_index))) {
                attn_fill += h * h + bulk;
            }
        }
        const auto & layer = model.layers[il];
        const double common = bytes({layer.attn_norm, layer.attn_post_norm, layer.attn_q_norm, layer.attn_k_norm, layer.ssm_norm});
        const double ff = bytes({layer.ffn_gate, layer.ffn_up, layer.ffn_down});
        const double mix = hp.is_recr(il) ? bytes({layer.wqkv, layer.ssm_conv1d, layer.ssm_beta, layer.ssm_alpha, layer.ssm_dt, layer.ssm_a}) :
                bytes({layer.wk, layer.wv, layer.wqkv});
        const double rest = hp.is_recr(il) ? bytes({layer.wqkv_gate, layer.ssm_out}) : bytes({layer.wq, layer.wo});
        const double full_bytes = common + ff + mix + rest;
        stock_bytes += full_bytes;
        if (il < uint32_t(params.split)) {
            lower_bytes += full_bytes;
        } else if (!layerwise || il == uint32_t(layer_index)) {
            const double map = params.projector ? bytes({params.projector->w.at(il), params.projector->b.at(il)}) : 4 * h * h;
            late_bytes += full_bytes;
            fill_bytes += map + common + mix;
            extra_bytes += ff + rest;
            if (!hp.is_recr(il)) { attn_bytes += map + common + mix; }
        }
        const double full = linear + mlp + attention;
        stock += full;
        if (il < uint32_t(params.split)) {
            lower += full;
        } else if (!layerwise || il == uint32_t(layer_index)) {
            late += full;
            // The current late graph projects every row, including exact rows.
            fill += h * h + bulk;
            extra += full - bulk;
        }
    }
    double work = 0, traffic = 0;
    switch (next) {
        case phase::lower:       work = count * lower; traffic = lower_bytes; break;
        case phase::s1_fill:     work = count * attn_fill; traffic = attn_bytes; break;
        case phase::s1_probe:    work = count * late; traffic = late_bytes; break;
        case phase::prefix:      work = count * fill + exact * extra; traffic = fill_bytes + (exact > 0 ? extra_bytes : 0); break;
        case phase::stream_tail:
        case phase::tail:
            work = count * late;
            traffic = late_bytes;
            if (params.vtail && position == params.n_prefix) { work += cost.vtail_rows * attn_fill; traffic += attn_bytes; }
            break;
        case phase::mass:
            for (int32_t il : layers) {
                const double q = hp.n_head(il) * hp.n_embd_head_k(il);
                work += (win_end - win_begin) * (2 * h * q + q * params.n_tokens);
                traffic += bytes({model.layers[il].wq, model.layers[il].attn_norm});
            }
            break;
        default: break;
    }
    // Charge compute and weight reads separately; sparse rows still read the full late FFN weights.
    const double microbatch = std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));
    const double compute = work / stock;
    const double weights = microbatch * traffic / stock_bytes;
    // Keep a cheaper unchanged graph within its stock physical-row cost, including n_batch == n_ubatch.
    const double charged = std::ceil(std::min(compute + weights, std::max({double(count), compute, weights})));
    if (!std::isfinite(charged) || charged > INT32_MAX) { throw std::runtime_error("RidgeFill budget: invalid step cost"); }
    const int32_t result = std::max(1, int32_t(charged));
    if (count > capacity || result > capacity) {
        throw std::runtime_error("RidgeFill: sweep step exceeds the stock prompt chunk budget; reduce probe/vtail rows or increase n_batch");
    }
    if (profiling) { llama_ridgefill_profile(params.projector, "budget_estimate", stream_layer, position, begin, ggml_time_us()); }
    return result;
}

void server_ridgefill_sweep::sweep_prefix(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        const server_ridgefill_sweep_params & params) {
    auto * maps = ctx->get_ridgefill_stream();
    if (position == 0 && stream_layer < 0) {
        auto part = llama_ridgefill_part_default_params();
        part.part = LLAMA_RIDGEFILL_PART_LATE;
        part.h_buf = hidden->tensor();
        if (llama_ridgefill_seq_set_rows(ctx, seq, rows.data(), rows.size()) < 0) {
            throw std::runtime_error("RidgeFill quality+qattn: exact rows were refused");
        }
#if defined(LLAMA_RIDGEFILL_API_VERSION) && LLAMA_RIDGEFILL_API_VERSION >= 2
        if (params.vtail) {
            if (llama_ridgefill_seq_set_vtail(ctx, seq, true, params.vtail_rows) < 0) {
                throw std::runtime_error("RidgeFill vtail: enabling value rescaling was refused");
            }
            vtail_enabled = true;
        }
#endif
        if (llama_ridgefill_seq_set_part(ctx, seq, &part) < 0) {
            throw std::runtime_error("RidgeFill quality+qattn: LATE part was refused");
        }
        if (maps) {
            stream_calib_rows = params.vtail ? std::max(cost.vtail_rows, std::min({params.n_batch, int32_t(llama_n_ubatch(ctx)),
                    params.n_tokens - params.n_prefix - int(params.defer_last_token)})) : 0;
            compact_hidden_rows(ctx, seq, params);
            stream_carry = std::make_unique<llama_ridgefill_sweep_buffer>(ctx->get_model().select_buft(params.split), params.n_embd,
                    std::max<int32_t>(1, rows.size() + stream_calib_rows), GGML_TYPE_F32, params.max_bytes);
            std::vector<float> seed(size_t(params.n_embd)*rows.size());
            for (size_t i = 0; i < rows.size(); ++i) {
                const auto * source = static_cast<const uint8_t *>(hidden->tensor()->data) + size_t(rows[i])*hidden->tensor()->nb[1];
                float * target = seed.data() + i*params.n_embd;
                if (params.type == GGML_TYPE_F16) { ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(source), target, params.n_embd); }
                else if (params.type == GGML_TYPE_BF16) { ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(source), target, params.n_embd); }
                else { std::memcpy(target, source, size_t(params.n_embd)*sizeof(float)); }
            }
            if (!seed.empty()) {
                ggml_backend_tensor_set(stream_carry->tensor(), seed.data(), 0, seed.size()*sizeof(float));
                if (!ggml_backend_buffer_is_host(stream_carry->tensor()->buffer)) { hidden_upload_bytes += seed.size()*sizeof(float); }
            }
            stream_layer = params.split;
            ctx->ridgefill_seq_stream_layer(seq, stream_layer, stream_carry->tensor(), params.n_prefix + stream_calib_rows);
            LLAMA_LOG_INFO("%s: RidgeFill stream compact residual %.4f MiB (%zu exact + %d tail rows), maps %.4f MiB\n", __func__,
                    stream_carry->size_bytes()/(1024.0*1024.0), rows.size(), stream_calib_rows, maps->buffer_bytes()/(1024.0*1024.0));
        }
    }
    if (maps) { maps->stage(*ctx, stream_layer, stream_layer + 1 < int32_t(ctx->get_model().hparams.n_layer()) ? stream_layer + 1 : -1); }
    decode(ctx, seq, tokens, std::min({params.n_batch, int32_t(llama_n_ubatch(ctx)), params.n_prefix - position}), false, false);
    if (maps && position == params.n_prefix) {
        if (stream_calib_rows) { next = phase::stream_tail; }
        else { advance_stream(ctx, seq, params, false); }
    }
}

void server_ridgefill_sweep::prefetch_hidden(llama_context * ctx, const server_ridgefill_sweep_params & params) {
    if (!ctx->get_ridgefill_stream() || ggml_backend_buffer_is_host(hidden->tensor()->buffer)) { return; }
    auto * source = hidden->tensor();
    hidden_readback = std::async(std::launch::async, [source, params]() {
        const bool profiling = params.projector && params.projector->profiling;
        const int64_t begin = profiling ? ggml_time_us() : 0;
        auto host = std::make_unique<llama_ridgefill_sweep_buffer>(ggml_backend_cpu_buffer_type(),
                params.n_embd, params.n_tokens, params.type, params.max_bytes);
        ggml_backend_tensor_get(source, host->tensor()->data, 0, host->size_bytes());
        if (profiling) { llama_ridgefill_profile(params.projector, "download_worker", -1, 0, begin, ggml_time_us()); }
        return host;
    });
}

void server_ridgefill_sweep::compact_hidden_rows(llama_context * ctx, llama_seq_id seq, const server_ridgefill_sweep_params & params) {
    const bool profiling = params.projector && params.projector->profiling;
    int64_t started = profiling ? ggml_time_us() : 0;
    auto profile = [&](const char * phase) {
        if (profiling) {
            const int64_t end = ggml_time_us();
            llama_ridgefill_profile(params.projector, phase, stream_layer, 0, started, end);
            started = end;
        }
    };
    ctx->synchronize();
    profile("compact_wait");
    std::unique_ptr<llama_ridgefill_sweep_buffer> host;
    if (hidden_readback.valid()) {
        host = hidden_readback.get();
    } else {
        host = std::make_unique<llama_ridgefill_sweep_buffer>(ggml_backend_cpu_buffer_type(), params.n_embd, params.n_tokens, params.type, params.max_bytes);
        profile("compact_host_alloc");
        ggml_backend_tensor_get(hidden->tensor(), host->tensor()->data, 0, hidden->size_bytes());
    }
    profile("compact_download");
    if (!ggml_backend_buffer_is_host(hidden->tensor()->buffer)) { hidden_download_bytes += hidden->size_bytes(); }
    ctx->ridgefill_seq_replace_hidden(seq, host->tensor());
    hidden = std::move(host);
    windows.clear();
    hidden_keep_end = params.n_prefix + stream_calib_rows;
    hidden_indices.assign(params.n_tokens, -1);
    const int32_t n_compact = hidden_keep_end - rows.size() + 1;
    const size_t row_bytes = hidden->tensor()->nb[1];
    std::vector<uint8_t> packed(llama_ridgefill_sweep_buffer::required_bytes(params.n_embd, n_compact, params.type, params.max_bytes), 0);
    size_t exact = 0;
    int32_t target = 1;
    for (int32_t pos = 0; pos < hidden_keep_end; ++pos) {
        if (exact < rows.size() && rows[exact] == pos) { hidden_indices[pos] = 0; ++exact; }
        else {
            hidden_indices[pos] = target;
            std::memcpy(packed.data() + size_t(target++)*row_bytes, static_cast<const uint8_t *>(hidden->tensor()->data) + size_t(pos)*row_bytes, row_bytes);
        }
    }
    if (target != n_compact || exact != rows.size()) { throw std::runtime_error("RidgeFill compact: incomplete exact row coverage"); }
    profile("compact_pack");
    const auto buft = ctx->get_model().select_buft(params.split);
    compact_hidden = std::make_unique<llama_ridgefill_sweep_buffer>(buft, params.n_embd, n_compact, params.type, params.max_bytes);
    profile("compact_device_alloc");
    ggml_backend_tensor_set(compact_hidden->tensor(), packed.data(), 0, packed.size());
    profile("compact_upload");
    if (!ggml_backend_buffer_is_host(compact_hidden->tensor()->buffer)) { hidden_upload_bytes += packed.size(); }
    tail_hidden = std::make_unique<llama_ridgefill_sweep_buffer>(buft, params.n_embd, llama_n_ubatch(ctx), params.type, params.max_bytes);
    ctx->ridgefill_seq_compact_hidden(seq, compact_hidden->tensor(), tail_hidden->tensor(), hidden_keep_end, &hidden_indices);
    profile("compact_finish");
    LLAMA_LOG_INFO("%s: RidgeFill hS host %.4f MiB, compact %.4f MiB, tail staging %.4f MiB\n", __func__,
            hidden->size_bytes()/1048576.0, compact_hidden->size_bytes()/1048576.0, tail_hidden->size_bytes()/1048576.0);
}

bool server_ridgefill_sweep::advance_stream(llama_context * ctx, llama_seq_id seq,
        const server_ridgefill_sweep_params & params, bool attn_only) {
    int32_t following = stream_layer + 1;
    const int32_t layers = ctx->get_model().hparams.n_layer();
    while (attn_only && following < layers && ctx->get_model().hparams.is_recr(following)) { ++following; }
    if (following == layers) {
        ctx->ridgefill_seq_stream_layer(seq, -1, nullptr, 0);
        stream_layer = -1;
        stream_carry.reset();
        return true;
    }
    stream_layer = following;
    ctx->ridgefill_seq_stream_layer(seq, stream_layer, attn_only ? nullptr : stream_carry->tensor(),
            params.n_prefix + (attn_only ? 0 : stream_calib_rows));
    position = 0;
    return false;
}

void server_ridgefill_sweep::sweep_tail(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        const server_ridgefill_sweep_params & params) {
    const int32_t end = params.n_tokens - int(params.defer_last_token);
    const bool streaming = next == phase::stream_tail;
    const bool last_layer = !streaming || stream_layer + 1 == int32_t(ctx->get_model().hparams.n_layer());
    if (position < end) {
        int32_t count = std::min({params.n_batch, int32_t(llama_n_ubatch(ctx)), end - position});
        if (streaming) {
            count = stream_calib_rows;
            ctx->get_ridgefill_stream()->stage(*ctx, stream_layer, last_layer ? -1 : stream_layer + 1);
        }
        const bool calibrate = params.vtail && position == params.n_prefix;
        if (calibrate) { count = std::max(count, cost.vtail_rows); }
#if defined(LLAMA_RIDGEFILL_API_VERSION) && LLAMA_RIDGEFILL_API_VERSION >= 2
        if (calibrate && (!streaming || stream_layer == params.split)) {
            const auto & cache = sweep_cache(ctx);
            cost.vtail_cache_rows = sweep_cache_rows(cache, seq);
            const auto & hp = ctx->get_model().hparams;
            for (uint32_t il = params.split; il < hp.n_layer(); ++il) {
                if (hp.is_recr(il)) { continue; }
                const uint64_t elements = uint64_t(cost.vtail_cache_rows) * hp.n_head_kv(il) * hp.n_embd_head_v(il);
                cost.vtail_cache_elements += elements;
                cost.vtail_cache_read_bytes += ggml_row_size(cache.type_v(), elements);
            }
            cost.vtail_cache_write_bytes = cost.vtail_cache_read_bytes;
        }
#endif
        decode(ctx, seq, tokens, count, params.all_tail_logits && last_layer,
                params.logits_begin < 0 && !params.defer_last_token && position + count == end && last_layer,
                last_layer ? params.logits_begin : -1, last_layer ? params.logits_end : -1);
#if defined(LLAMA_RIDGEFILL_API_VERSION) && LLAMA_RIDGEFILL_API_VERSION >= 2
        if (calibrate && last_layer && llama_ridgefill_seq_get_vtail_scales(ctx, seq, vtail_scales.data(), vtail_scales.size()) != int32_t(vtail_scales.size())) {
            throw std::runtime_error("RidgeFill vtail: complete per-layer value scales are unavailable");
        }
        if (calibrate && last_layer && !std::all_of(vtail_scales.begin(), vtail_scales.end(), [](float scale) { return std::isfinite(scale) && scale >= 0; })) {
            throw std::runtime_error("RidgeFill vtail: nonfinite or negative value scale");
        }
#endif
    }
    if (streaming) {
        if (!advance_stream(ctx, seq, params, false)) { next = phase::prefix; return; }
        next = phase::tail;
    }
    if (position == end && !params.defer_last_token) {
        finish_external_tail(ctx, seq);
    }
}

void server_ridgefill_sweep::finish_external_tail(llama_context * ctx, llama_seq_id seq) {
    llama_synchronize(ctx);
    if (hidden_readback.valid()) {
        auto readback = hidden_readback.get();
        hidden_download_bytes += readback->size_bytes();
    }
    if (auto * maps = ctx->get_ridgefill_stream()) {
        LLAMA_LOG_INFO("%s: RidgeFill context map uploads total %" PRIu64 " bytes\n", __func__, maps->uploaded_bytes());
        LLAMA_LOG_INFO("%s: RidgeFill hS link uploaded %" PRIu64 " downloaded %" PRIu64 " bytes\n", __func__, hidden_upload_bytes, hidden_download_bytes);
    }
#if defined(LLAMA_RIDGEFILL_API_VERSION) && LLAMA_RIDGEFILL_API_VERSION >= 2
    if (vtail_enabled && llama_ridgefill_seq_set_vtail(ctx, seq, false, 1) < 0) {
        throw std::runtime_error("RidgeFill vtail: disabling value rescaling before FULL was refused");
    }
#endif
    vtail_enabled = false;
    if (llama_ridgefill_seq_set_part(ctx, seq, nullptr) < 0) {
        throw std::runtime_error("RidgeFill quality+qattn: completed replay could not reset FULL");
    }
    active = false;
    hidden.reset();
    stream_carry.reset();
    compact_hidden.reset();
    tail_hidden.reset();
    windows.clear();
    next = phase::done;
}

void server_ridgefill_sweep::cancel(llama_context * ctx, llama_seq_id seq) {
    if (active) {
        llama_synchronize(ctx);
        llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
        finish_external_tail(ctx, seq);
    }
}

server_ridgefill_sweep_result server_ridgefill_sweep_step(
        server_ridgefill_sweep & sweep,
        llama_context * ctx,
        llama_seq_id seq,
        const llama_token * tokens,
        const std::vector<llama_pos> & base_rows,
        const server_ridgefill_sweep_params & params) {
    if (!std::isfinite(params.qattn_share) || params.qattn_share < 0.0f || params.qattn_share > 1.0f) {
        throw std::runtime_error("RidgeFill quality+qattn: qattn share must be finite and in [0, 1]");
    }

    if (params.lexical && params.qattn_share == 0) {
        throw std::runtime_error("RidgeFill lexical: requires positive question-attention share for its exact-row budget");
    }
    if (params.qattn_share == 0.0f && !params.s1 && !params.vtail && !params.lexical) {
        return server_ridgefill_sweep_result::quality;
    }
    if (params.n_tokens <= 0 || params.n_prefix < 0 || params.n_prefix >= params.n_tokens) {
        throw std::runtime_error("RidgeFill quality+qattn: prefix must be within a nonempty prompt with an exact tail");
    }
    if (params.n_prefix == 0) {
        return server_ridgefill_sweep_result::quality;
    }
    if ((params.s1 || params.vtail) && server_ridgefill_sweep_api_version() < 2) {
        throw std::runtime_error("RidgeFill S1/vtail: libllama C API v2 is required");
    }
    const int32_t end = params.qattn_end >= 0 ? params.qattn_end : params.n_tokens;
    if (params.s1 && (!params.projector || params.qattn_blind || params.s1_window <= 0 ||
            params.s1_window > end - params.n_prefix || !std::isfinite(params.s1_factor) || params.s1_factor < 0)) {
        throw std::runtime_error("RidgeFill S1: requires a projector, nonnegative finite factor, and a nonempty window entirely in the known exact tail");
    }
    if (params.vtail && params.vtail_rows <= 0) {
        throw std::runtime_error("RidgeFill vtail: calibration rows must be positive");
    }
    if (params.n_batch <= 0 || params.qattn_window <= 0) {
        throw std::runtime_error("RidgeFill quality+qattn: batch and question window must be positive");
    }
    if (params.qattn_window_mode != "last" && params.qattn_window_mode != "turn") {
        throw std::runtime_error("RidgeFill quality+qattn: qattn-window-mode must be last or turn");
    }
    if (params.qattn_turn_cap <= 0) {
        throw std::runtime_error("RidgeFill quality+qattn: qattn-turn-cap must be positive");
    }
    if (params.qattn_window_mode == "turn" && params.qattn_blind) {
        throw std::runtime_error("RidgeFill qattn-window-mode turn: blind selection is unsupported");
    }
    if (params.qattn_end < -1 || params.qattn_end == 0 || params.qattn_end > params.n_tokens ||
            (params.qattn_end >= 0 && params.qattn_blind)) {
        throw std::runtime_error("RidgeFill quality+qattn: invalid or conflicting question-window boundary");
    }
    if ((params.logits_begin != -1 || params.logits_end != -1) &&
            (params.logits_begin < params.n_prefix || params.logits_end <= params.logits_begin ||
             params.logits_end > params.n_tokens || params.defer_last_token || params.all_tail_logits)) {
        throw std::runtime_error("RidgeFill quality+qattn: scoring logits must be a nonempty exact-tail range without deferred output");
    }
    llama_ridgefill_sweep_buffer::required_bytes(params.n_embd, params.n_tokens, params.type, params.max_bytes);

    if (!ctx || !tokens) {
        throw std::runtime_error("RidgeFill quality+qattn: missing context or prompt tokens");
    }
    if (params.vtail) {
        const int32_t needed = std::min(params.vtail_rows, params.n_tokens - params.n_prefix);
        if (llama_n_ubatch(ctx) < uint32_t(needed) || llama_n_batch(ctx) < uint32_t(needed)) {
            throw std::runtime_error("RidgeFill vtail: context batch and ubatch must hold the first calibration rows");
        }
        if (params.defer_last_token && needed == params.n_tokens - params.n_prefix) {
            throw std::runtime_error("RidgeFill vtail: calibration cannot consume the server's deferred final token; reduce vtail rows");
        }
    }
    sweep.n_outputs = 0;
    sweep.n_step_tokens = 0;
    sweep.output_position = -1;
    using phase = server_ridgefill_sweep::phase;
    switch (sweep.next) {
        case phase::lower:
            sweep.sweep_lower(ctx, seq, tokens, params);
            if (sweep.position == params.n_tokens) {
                sweep.next = params.qattn_share > 0 ? phase::mass : phase::rows;
            }
            break;
        case phase::mass:
            sweep.question_mass(ctx, seq, params);
            sweep.next = phase::rows;
            break;
        case phase::rows:
            sweep.select_rows(ctx, tokens, base_rows, params);
            sweep.position = 0;
            sweep.next = params.s1 ? phase::s1_fill : phase::prefix;
            break;
        case phase::s1_fill:
        case phase::s1_probe:
        case phase::s1_rows:
#if defined(LLAMA_RIDGEFILL_API_VERSION) && LLAMA_RIDGEFILL_API_VERSION >= 2
            if (sweep.next == phase::s1_fill) {
                sweep.sweep_s1_fill(ctx, seq, tokens, params);
                if (sweep.position == params.n_prefix) {
                    sweep.position = sweep.win_end - params.s1_window;
                    sweep.next = phase::s1_probe;
                }
            } else if (sweep.next == phase::s1_probe) {
                sweep.sweep_s1_probe(ctx, seq, tokens, params);
                if (sweep.position == sweep.win_end) { sweep.next = phase::s1_rows; }
            } else {
                sweep.select_s1_rows(ctx, seq, tokens, params);
                sweep.position = 0;
                sweep.next = phase::prefix;
            }
            break;
#else
            throw std::runtime_error("RidgeFill S1: libllama C API v2 is required");
#endif
        case phase::prefix:
            sweep.sweep_prefix(ctx, seq, tokens, params);
            if (sweep.next == phase::prefix && sweep.position == params.n_prefix) {
                sweep.next = phase::tail;
            }
            break;
        case phase::stream_tail:
        case phase::tail:
            sweep.sweep_tail(ctx, seq, tokens, params);
            if (params.defer_last_token && sweep.position == params.n_tokens - 1) {
                if (sweep.tail_hidden) {
                    ctx->ridgefill_seq_stage_tail(seq, sweep.position, 1);
                    if (!ggml_backend_buffer_is_host(sweep.tail_hidden->tensor()->buffer)) { sweep.hidden_upload_bytes += sweep.hidden->tensor()->nb[1]; }
                }
                sweep.next = phase::external;
                return server_ridgefill_sweep_result::ready;
            }
            break;
        case phase::external:
            return server_ridgefill_sweep_result::ready;
        case phase::done:
            return server_ridgefill_sweep_result::complete;
    }
    return sweep.next == phase::done ? server_ridgefill_sweep_result::complete : server_ridgefill_sweep_result::pending;
}
