#include "server-ridgefill-sweep.h"

#include "../../src/llama-ridgefill-sweep.h"
#include "../../src/llama-context.h"
#include "../../src/llama-model.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

server_ridgefill_sweep::server_ridgefill_sweep() = default;
server_ridgefill_sweep::~server_ridgefill_sweep() = default;

void server_ridgefill_sweep::decode(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        int32_t count, bool output_all, bool output_last, int32_t logits_begin, int32_t logits_end) {
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
}

void server_ridgefill_sweep::sweep_lower(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        const server_ridgefill_sweep_params & params) {
    const auto & model = ctx->get_model();
    if (!hidden) {
        if (params.split <= 0 || params.split >= (int32_t) model.hparams.n_layer() || params.n_embd != (int32_t) model.hparams.n_embd) {
            throw std::runtime_error("RidgeFill quality+qattn: invalid split or embedding width");
        }
        win_end = params.qattn_end >= 0 ? params.qattn_end : (params.qattn_blind ? params.n_prefix : params.n_tokens);
        win_begin = std::max(0, win_end - params.qattn_window);
        if (params.qattn_window_mode == "turn") {
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
        for (int32_t il : {3, 7, 11, 15}) {
            if (il >= params.split) {
                continue;
            }
            if (model.hparams.is_recr(il)) {
                throw std::runtime_error("RidgeFill quality+qattn: question layer is not an attention layer");
            }
            layers.push_back(il);
            windows.push_back(std::make_unique<llama_ridgefill_sweep_buffer>(model.select_buft(il),
                    params.n_embd, win_end - win_begin, GGML_TYPE_F32));
        }
        if (layers.empty()) {
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
        part.win_begin = win_begin;
        part.win_n = win_end - win_begin;
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
    std::vector<ggml_tensor *> buffers;
    for (const auto & window : windows) { buffers.push_back(window->tensor()); }
    mass = llama_ridgefill_sweep_question_mass(*ctx, seq, params.n_tokens, win_end, layers, buffers);
}

void server_ridgefill_sweep::select_rows(const std::vector<llama_pos> & base_rows, const server_ridgefill_sweep_params & params) {
    if (mass.size() != size_t(params.n_tokens)) {
        throw std::runtime_error("RidgeFill quality+qattn: question mass does not cover the prompt");
    }
    if (!std::is_sorted(base_rows.begin(), base_rows.end()) ||
            std::adjacent_find(base_rows.begin(), base_rows.end()) != base_rows.end() ||
            (!base_rows.empty() && (base_rows.front() < 0 || base_rows.back() >= params.n_prefix))) {
        throw std::runtime_error("RidgeFill quality+qattn: base exact rows must be sorted unique prefix positions");
    }
    const auto question_rows = llama_ridgefill_sweep_top_rows(mass, params.n_prefix, params.qattn_share);
    rows.clear();
    std::set_union(base_rows.begin(), base_rows.end(), question_rows.begin(), question_rows.end(), std::back_inserter(rows));
}

void server_ridgefill_sweep::sweep_prefix(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        const server_ridgefill_sweep_params & params) {
    if (position == 0) {
        auto part = llama_ridgefill_part_default_params();
        part.part = LLAMA_RIDGEFILL_PART_LATE;
        part.h_buf = hidden->tensor();
        if (llama_ridgefill_seq_set_rows(ctx, seq, rows.data(), rows.size()) < 0 ||
                llama_ridgefill_seq_set_part(ctx, seq, &part) < 0) {
            throw std::runtime_error("RidgeFill quality+qattn: exact rows or LATE part were refused");
        }
    }
    decode(ctx, seq, tokens, std::min({params.n_batch, int32_t(llama_n_ubatch(ctx)), params.n_prefix - position}), false, false);
}

void server_ridgefill_sweep::sweep_tail(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        const server_ridgefill_sweep_params & params) {
    const int32_t end = params.n_tokens - int(params.defer_last_token);
    if (position < end) {
        const int32_t count = std::min({params.n_batch, int32_t(llama_n_ubatch(ctx)), end - position});
        decode(ctx, seq, tokens, count, params.all_tail_logits,
                params.logits_begin < 0 && !params.defer_last_token && position + count == end, params.logits_begin, params.logits_end);
    }
    if (position == end && !params.defer_last_token) {
        finish_external_tail(ctx, seq);
    }
}

void server_ridgefill_sweep::finish_external_tail(llama_context * ctx, llama_seq_id seq) {
    llama_synchronize(ctx);
    if (llama_ridgefill_seq_set_part(ctx, seq, nullptr) < 0) {
        throw std::runtime_error("RidgeFill quality+qattn: completed replay could not reset FULL");
    }
    active = false;
    hidden.reset();
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
    if (params.qattn_share == 0.0f) {
        return server_ridgefill_sweep_result::quality;
    }
    if (params.n_tokens <= 0 || params.n_prefix < 0 || params.n_prefix >= params.n_tokens) {
        throw std::runtime_error("RidgeFill quality+qattn: prefix must be within a nonempty prompt with an exact tail");
    }
    if (params.n_prefix == 0) {
        return server_ridgefill_sweep_result::quality;
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
    sweep.n_outputs = 0;
    sweep.output_position = -1;
    using phase = server_ridgefill_sweep::phase;
    switch (sweep.next) {
        case phase::lower:
            sweep.sweep_lower(ctx, seq, tokens, params);
            if (sweep.position == params.n_tokens) {
                sweep.next = phase::mass;
            }
            break;
        case phase::mass:
            sweep.question_mass(ctx, seq, params);
            sweep.next = phase::rows;
            break;
        case phase::rows:
            sweep.select_rows(base_rows, params);
            sweep.position = 0;
            sweep.next = phase::prefix;
            break;
        case phase::prefix:
            sweep.sweep_prefix(ctx, seq, tokens, params);
            if (sweep.position == params.n_prefix) {
                sweep.next = phase::tail;
            }
            break;
        case phase::tail:
            sweep.sweep_tail(ctx, seq, tokens, params);
            if (params.defer_last_token && sweep.position == params.n_tokens - 1) {
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
