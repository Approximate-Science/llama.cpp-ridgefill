#include "server-ridgefill-sweep.h"

#include "../../src/llama-ridgefill-sweep.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

server_ridgefill_sweep::server_ridgefill_sweep() = default;
server_ridgefill_sweep::~server_ridgefill_sweep() = default;

void server_ridgefill_sweep::sweep_lower(llama_context *, llama_seq_id, const llama_token *, const server_ridgefill_sweep_params &) {
    throw std::runtime_error("RidgeFill quality+qattn: sweep 1 requires LOWER dispatch and llama_ridgefill_seq_set_part (unavailable)");
}

void server_ridgefill_sweep::question_mass(llama_context *, llama_seq_id, const server_ridgefill_sweep_params &) {
    throw std::runtime_error("RidgeFill quality+qattn: question mass requires captured LOWER layer inputs and cache rows (unavailable)");
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

void server_ridgefill_sweep::sweep_prefix(llama_context *, llama_seq_id, const llama_token *, const server_ridgefill_sweep_params &) {
    throw std::runtime_error("RidgeFill quality+qattn: sweep 2 requires LATE fill dispatch with preserved lower state and KV cells (unavailable)");
}

void server_ridgefill_sweep::sweep_tail(llama_context *, llama_seq_id, const llama_token *, const server_ridgefill_sweep_params &) {
    throw std::runtime_error("RidgeFill quality+qattn: sweep 3 requires LATE exact dispatch with tail logits (unavailable)");
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
            if (sweep.position == params.n_tokens) {
                sweep.next = phase::done;
            }
            break;
        case phase::done:
            return server_ridgefill_sweep_result::complete;
    }
    return sweep.next == phase::done ? server_ridgefill_sweep_result::complete : server_ridgefill_sweep_result::pending;
}
