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
        int32_t count, bool output_all, bool output_last) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int32_t i = 0; i < count; ++i) {
        batch.token[i] = tokens[position + i];
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = output_all || (output_last && i == count - 1);
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("RidgeFill quality+qattn: sweep decode failed with code " + std::to_string(rc));
    }
    n_outputs = output_all ? count : int(output_last);
    position += count;
}

void server_ridgefill_sweep::sweep_lower(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        const server_ridgefill_sweep_params & params) {
    const auto & model = ctx->get_model();
    const int32_t end = params.qattn_blind ? params.n_prefix : params.n_tokens;
    const int32_t start = std::max(0, end - params.qattn_window);
    if (!hidden) {
        if (params.split <= 0 || params.split >= (int32_t) model.hparams.n_layer() || params.n_embd != (int32_t) model.hparams.n_embd) {
            throw std::runtime_error("RidgeFill quality+qattn: invalid split or embedding width");
        }
        for (int32_t il : {3, 7, 11, 15}) {
            if (il >= params.split) {
                continue;
            }
            if (model.hparams.is_recr(il)) {
                throw std::runtime_error("RidgeFill quality+qattn: question layer is not an attention layer");
            }
            layers.push_back(il);
            windows.emplace_back(size_t(end - start) * params.n_embd);
        }
        if (layers.empty()) {
            throw std::runtime_error("RidgeFill quality+qattn: split has no lower attention layers");
        }
        hidden = std::make_unique<llama_ridgefill_sweep_buffer>(model.select_buft(params.split),
                params.n_embd, params.n_tokens, params.type, params.max_bytes);
        if (llama_ridgefill_seq_set_part(ctx, seq, LLAMA_RIDGEFILL_PART_LOWER, hidden->tensor()) < 0) {
            throw std::runtime_error("RidgeFill quality+qattn: LOWER part was refused");
        }
        active = true;
    }
    const int32_t count = std::min({params.n_batch, int32_t(llama_n_ubatch(ctx)), params.n_tokens - position});
    const int32_t first = position;
    const bool capture = first < end && first + count > start;
    std::vector<int32_t> enabled;
    if (capture) {
        for (int32_t il : layers) {
            if (!ctx->get_cparams().embeddings_layer_inp[il]) {
                llama_set_embeddings_layer_inp(ctx, il, true);
                enabled.push_back(il);
            }
        }
    }
    try {
        decode(ctx, seq, tokens, count, false, false);
        n_lower = position;
        if (capture) {
            const int32_t from = std::max(first, start);
            const int32_t to = std::min(position, end);
            for (size_t i = 0; i < layers.size(); ++i) {
                const auto * input = llama_get_embeddings_layer_inp(ctx, layers[i]);
                if (!input) {
                    throw std::runtime_error("RidgeFill quality+qattn: LOWER query input is missing");
                }
                std::copy(input + size_t(from - first) * params.n_embd, input + size_t(to - first) * params.n_embd,
                        windows[i].data() + size_t(from - start) * params.n_embd);
            }
        }
    } catch (...) {
        for (int32_t il : enabled) { llama_set_embeddings_layer_inp(ctx, il, false); }
        throw;
    }
    for (int32_t il : enabled) { llama_set_embeddings_layer_inp(ctx, il, false); }
}

void server_ridgefill_sweep::question_mass(llama_context * ctx, llama_seq_id seq, const server_ridgefill_sweep_params & params) {
    mass = llama_ridgefill_sweep_question_mass(*ctx, seq, params.n_tokens,
            params.qattn_blind ? params.n_prefix : params.n_tokens, layers, windows);
    windows.clear();
}

void server_ridgefill_sweep::select_rows(const std::vector<llama_pos> & base_rows, const server_ridgefill_sweep_params & params) {
    if (mass.size() != size_t(params.n_tokens)) {
        throw std::runtime_error("RidgeFill quality+qattn: question mass does not cover the prompt");
    }
    if (!std::is_sorted(base_rows.begin(), base_rows.end()) ||
            std::adjacent_find(base_rows.begin(), base_rows.end()) != base_rows.end() ||
            (!base_rows.empty() && (base_rows.front() < 0 || base_rows.back() >= params.n_prefix))) {
        throw std::runtime_error("RidgeFill quality+qattn: class and digit rows must be sorted unique prefix positions");
    }
    const auto question_rows = llama_ridgefill_sweep_top_rows(mass, params.n_prefix, params.qattn_share);
    rows.clear();
    std::set_union(base_rows.begin(), base_rows.end(), question_rows.begin(), question_rows.end(), std::back_inserter(rows));
}

void server_ridgefill_sweep::sweep_prefix(llama_context * ctx, llama_seq_id seq, const llama_token * tokens,
        const server_ridgefill_sweep_params & params) {
    if (position == 0) {
        if (llama_ridgefill_seq_set_rows(ctx, seq, rows.data(), rows.size()) < 0 ||
                llama_ridgefill_seq_set_part(ctx, seq, LLAMA_RIDGEFILL_PART_LATE, hidden->tensor()) < 0) {
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
        decode(ctx, seq, tokens, count, params.all_tail_logits, !params.defer_last_token && position + count == end);
    }
    if (position == end && !params.defer_last_token) {
        finish_external_tail(ctx, seq);
    }
}

void server_ridgefill_sweep::finish_external_tail(llama_context * ctx, llama_seq_id seq) {
    if (llama_ridgefill_seq_set_part(ctx, seq, LLAMA_RIDGEFILL_PART_FULL, nullptr) < 0) {
        throw std::runtime_error("RidgeFill quality+qattn: completed replay could not reset FULL");
    }
    active = false;
    hidden.reset();
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
    llama_ridgefill_sweep_buffer::required_bytes(params.n_embd, params.n_tokens, params.type, params.max_bytes);

    if (!ctx || !tokens) {
        throw std::runtime_error("RidgeFill quality+qattn: missing context or prompt tokens");
    }
    sweep.n_outputs = 0;
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
