#include "llama-model.h"
#include "llama-kv-cache.h"
#include "ggml-backend.h"
#include "llama-ridgefill.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

static std::atomic<size_t> allocations{0};
static std::atomic<bool> counting{false};

void * operator new(size_t size) {
    if (void * p = std::malloc(size ? size : 1)) {
        if (counting.load(std::memory_order_relaxed)) { allocations.fetch_add(1, std::memory_order_relaxed); }
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }

static void check_rows() {
    std::vector<llama_pos> selected[2] = {{-2, 0, 2, 2, 4, 7, 9, 31}, {1, 3, 8}};
    std::vector<llama_pos> positions(8);
    std::vector<int32_t> counts(8, 1);
    llama_seq_id seq = 0, other = 1;
    std::vector<llama_seq_id *> owners(8, &seq);
    llama_ubatch b = {};
    b.n_tokens = b.n_seq_tokens = 8; b.n_seqs = b.n_seqs_unq = b.n_pos = 1;
    b.pos = positions.data(); b.n_seq_id = counts.data(); b.seq_id = owners.data(); b.seq_id_unq = &seq;
    int cases = 0;
    auto compare = [&]() {
        std::vector<int32_t> expected, actual{-1, -2};
        for (uint32_t i = 0; i < b.n_tokens; ++i) {
            const auto & rows = selected[*owners[i]];
            if (std::find(rows.begin(), rows.end(), positions[i]) != rows.end()) { expected.push_back(i); }
        }
        if (llama_ridgefill_ubatch_rows(b, selected, &actual) != int(expected.size()) || actual != expected ||
                llama_ridgefill_ubatch_rows(b, selected, nullptr) != int(expected.size())) {
            throw std::runtime_error("selected row indices differ from membership reference");
        }
        ++cases;
    };
    for (int i = 0; i < 8; ++i) { positions[i] = i; }
    compare();
    for (int i = 0; i < 8; ++i) { positions[i] = i - 2; }
    compare();
    positions = {0, 3, 2, 2, 7, 9, 31, INT32_MAX}; b.pos = positions.data(); compare();
    for (int i = 0; i < 8; ++i) { positions[i] = i; }
    owners[3] = &other; compare(); owners[3] = &seq;
    counts[2] = 2; compare(); counts[2] = 1;
    b.n_seqs_unq = 2; compare(); b.n_seqs_unq = 1;
    b.n_tokens = 0; compare(); b.n_tokens = 8;
    selected[0].clear(); compare();
    std::vector<int32_t> out{-1};
    if (llama_ridgefill_ubatch_rows(b, nullptr, &out) != 0 || !out.empty()) { throw std::runtime_error("missing selection did not clear indices"); }
    std::printf("row membership cases=%d\n", cases + 1);
}

int main(int argc, char ** argv) {
    try {
        if (argc != 2) { throw std::runtime_error("usage: test-replay-index TINY_MODEL"); }
        check_rows();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) { throw std::runtime_error("model load failed"); }
        auto hp = model->hparams;
        hp.rope_type = LLAMA_ROPE_TYPE_NORM;
        for (const uint32_t n : {16384u, 32768u}) {
            llama_kv_cache cache(*model, hp, GGML_TYPE_F32, GGML_TYPE_F32, false, false, true,
                    n, 1, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, [](uint32_t) { return false; }, nullptr, nullptr);
            std::vector<llama_pos> positions(n);
            std::vector<int32_t> counts(n, 1);
            llama_seq_id seq = 0;
            std::vector<llama_seq_id *> owners(n, &seq);
            for (uint32_t i = 0; i < n; ++i) { positions[i] = i; }
            llama_ubatch batch = {};
            batch.n_tokens = batch.n_seq_tokens = n;
            batch.n_seqs = batch.n_seqs_unq = batch.n_pos = 1;
            batch.pos = positions.data(); batch.n_seq_id = counts.data();
            batch.seq_id = owners.data(); batch.seq_id_unq = &seq;
            auto slot = cache.find_slot(batch, false);
            if (slot.empty()) { throw std::runtime_error("cannot seed replay cells"); }
            cache.apply_ubatch(slot, batch);
            batch.n_tokens = batch.n_seq_tokens = 512;
            allocations = 0;
            counting = true;
            const auto replay = cache.find_slot_replay(batch);
            counting = false;
            if (replay.empty() || replay.idxs[0].size() != 512) { throw std::runtime_error("replay cells missing"); }
            for (uint32_t i = 0; i < 512; ++i) {
                if (replay.idxs[0][i] != i) { throw std::runtime_error("replay cell differs"); }
            }
            std::printf("cells=%u queries=512 allocations=%zu\n", n, allocations.load());
            if (allocations > 32) { throw std::runtime_error("replay lookup allocates per occupied cell"); }
            counts[0] = 0; owners[0] = nullptr;
            if (!cache.find_slot_replay(batch).empty()) { throw std::runtime_error("replay accepted a row without an owner"); }
            counts[0] = 1; owners[0] = &seq;
            cache.seq_div(seq, 0, n, 2);
            const auto duplicate = cache.find_slot_replay(batch);
            if (duplicate.empty()) { throw std::runtime_error("duplicate replay positions missing"); }
            for (uint32_t i = 0; i < 512; ++i) {
                if (duplicate.idxs[0][i] != 2*i + 1) { throw std::runtime_error("duplicate replay did not select the highest cell"); }
            }
            cache.seq_rm(seq, 128, 129);
            if (!cache.find_slot_replay(batch).empty()) { throw std::runtime_error("replay accepted a hole in dense query positions"); }
        }
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        counting = false;
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
