// Derived from L-B's tests/ridgefill/rf_dump.cpp (rf27-tools e0e1fa9) with --n-rs-seq (recurrent rollback snapshots)
// and --cross (batches cross the fill end, as the server submits them; no prefix dump then), --fa 0|1,
// --n-seq N (the same prompt on N sequences of a unified KV cache, in the same batches; seq s dumps to <output>.s<s>)
// --select-special/--select-header (chat rules), --last-marker IDS, and --rows1 (exact rows of sequences >= 1; empty: speed),
// --sweep (the two-sweep prefill: LOWER over all rows into an h_S buffer, then a LATE replay over all rows; writes
// <output>.h.f32), --win-layers L,.. --win-n W (LOWER also captures those layers' inputs at the last W positions;
// <output>.win<L>.f32), --layer-inp L,.. (no sweep, one batch: upstream's layer-input extraction of every row, <output>.inp<L>.f32).
// API v2: --s1-window W --s1-k K [--s1-special 0|1] (sweep + S1: S1_FILL [0, P), S1_PROBE [N - W, N), rows = top K by
// the S1 mass (ties: lower position) united with the special rows; <output>.mass_s1.f64, <output>.rows.i32),
// --vtail C (vtail on the LATE pass, rows C; <output>.vtail.f32), --sweep-boundary (the LATE batches stop at P and the
// state there is dumped to <output>.prefix.kv, all cells), --s1-begin B (the probe window starts at B instead: refusal
// tests), --s1-late-early (LATE right after S1_FILL: refused).
// CPU gate harness. Dump prefix and final sequence state plus tail logits; never score a filled row.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void write_bytes(const std::string & path, const void * data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char *>(data), size);
    if (!file) {
        throw std::runtime_error("cannot write " + path);
    }
}

static void dump_state(llama_context * ctx, const std::string & path, llama_seq_id seq = 0) {
    const size_t size = llama_state_seq_get_size(ctx, seq);
    if (size == 0) {
        throw std::runtime_error("sequence state is empty");
    }
    std::vector<uint8_t> data(size);
    if (llama_state_seq_get_data(ctx, data.data(), data.size(), seq) != size) {
        throw std::runtime_error("sequence state size changed");
    }
    write_bytes(path, data.data(), data.size());
}

static std::vector<int> int_list(const std::string & a) {
    std::vector<int> r;
    size_t b = 0;
    while (b < a.size()) {
        size_t e = a.find(',', b);
        r.push_back(std::stoi(a.substr(b, e - b)));
        if (e == std::string::npos) { break; }
        b = e + 1;
    }
    return r;
}

static void write_tensor(const std::string & path, const ggml_tensor * t, int64_t rows) {
    std::vector<float> v(t->ne[0] * rows);
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    write_bytes(path, v.data(), v.size() * sizeof(float));
}

static int run(int argc, char ** argv) {
    std::string model_path, tokens_path, output, projector, mode = "off", rows_arg;
    int n_rs_seq = 0;
    bool cross = false;
    int fa = 0;
    int n_seq = 1;
    bool sweep = false;
    bool late_only = false;
    bool warm = false;
    bool lower_twice = false, full_midsweep = false;
    float select_share = -1.0f;
    int select_digits = 0;
    int select_special = 0, select_header = 0;
    std::string marker_arg;
    std::string win_arg, inp_arg;
    int win_n = 0;
    std::string rows1_arg;
    int tail = 200, keep = 200, n_batch = 700, n_ubatch = 512;
    float alpha = 0;
    bool filled_logit = false;
    int s1_window = 0, s1_k = 0, s1_special = 0, vtail = 0, s1_begin_arg = -1;
    bool sweep_boundary = false, s1_late_early = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() {
            if (++i >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return std::string(argv[i]);
        };
        if      (arg == "--model")     { model_path = next(); }
        else if (arg == "--tokens")    { tokens_path = next(); }
        else if (arg == "--output")    { output = next(); }
        else if (arg == "--projector") { projector = next(); }
        else if (arg == "--mode")      { mode = next(); }
        else if (arg == "--tail")      { tail = std::stoi(next()); }
        else if (arg == "--keep")      { keep = std::stoi(next()); }
        else if (arg == "--batch")     { n_batch = std::stoi(next()); }
        else if (arg == "--ubatch")    { n_ubatch = std::stoi(next()); }
        else if (arg == "--alpha")     { alpha = std::stof(next()); }
        else if (arg == "--rows")      { rows_arg = next(); }
        else if (arg == "--filled-logit") { filled_logit = true; }
        else if (arg == "--n-rs-seq")  { n_rs_seq = std::stoi(next()); }
        else if (arg == "--cross")     { cross = true; }
        else if (arg == "--fa")        { fa = std::stoi(next()); }
        else if (arg == "--n-seq")     { n_seq = std::stoi(next()); }
        else if (arg == "--rows1")     { rows1_arg = next(); }
        else if (arg == "--sweep")     { sweep = true; }
        else if (arg == "--lower-twice") { lower_twice = true; }  // set LOWER again after the pass (refused)
        else if (arg == "--full-midsweep") { full_midsweep = true; } // FULL right after LOWER (refused)
        else if (arg == "--select-share")  { select_share = std::stof(next()); } // write llama_ridgefill_select_rows of [0, n - tail) and exit
        else if (arg == "--select-digits") { select_digits = std::stoi(next()); }
        else if (arg == "--select-special") { select_special = std::stoi(next()); }
        else if (arg == "--select-header")  { select_header = std::stoi(next()); }
        else if (arg == "--last-marker")    { marker_arg = next(); } // ids: print llama_ridgefill_last_marker over the prompt and exit
        else if (arg == "--warm")      { warm = true; } // decode a reversed prompt first, then seq_rm (stale states in the cells)
        else if (arg == "--late-only") { sweep = true; late_only = true; } // a LATE replay without LOWER first (refused)
        else if (arg == "--win-layers") { win_arg = next(); }
        else if (arg == "--win-n")     { win_n = std::stoi(next()); }
        else if (arg == "--layer-inp") { inp_arg = next(); }
        else if (arg == "--s1-window") { s1_window = std::stoi(next()); }
        else if (arg == "--s1-k")      { s1_k = std::stoi(next()); }
        else if (arg == "--s1-special") { s1_special = std::stoi(next()); }
        else if (arg == "--s1-begin")  { s1_begin_arg = std::stoi(next()); }
        else if (arg == "--s1-late-early") { s1_late_early = true; }
        else if (arg == "--vtail")     { vtail = std::stoi(next()); }
        else if (arg == "--sweep-boundary") { sweep_boundary = true; }
        else { throw std::invalid_argument("unknown argument " + arg); }
    }
    if (model_path.empty() || tokens_path.empty() || output.empty() || n_batch < 1 || n_ubatch < 1 || tail < 1 || keep < 1 || !(alpha >= 0 && alpha <= 1)) {
        throw std::invalid_argument("model, tokens, output and positive batch/tail/keep are required; alpha must be in [0, 1]");
    }
    if (mode != "off" && mode != "speed" && mode != "quality") {
        throw std::invalid_argument("unsupported --mode " + mode + ": quality+qattn requires the sweep harness");
    }
#ifndef RF27_API
    if (alpha != 0 || mode == "quality") {
        throw std::invalid_argument("--alpha/--mode quality require the RF27 engine API");
    }
#endif
    std::ifstream input(tokens_path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0 || input.tellg() % sizeof(int32_t) != 0) {
        throw std::invalid_argument("--tokens must contain int32 token IDs");
    }
    std::vector<int32_t> tokens(static_cast<size_t>(input.tellg()) / sizeof(int32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(tokens.data()), tokens.size() * sizeof(int32_t));
    const int n = tokens.size();
    const int prefix = mode == "off" || tail >= n ? 0 : n - tail;
    if (keep > n - prefix || (mode != "off" && projector.empty())) {
        throw std::invalid_argument("keep exceeds exact tail or mode requires --projector");
    }
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
    if (!model) { throw std::runtime_error("model load failed"); }
    std::unique_ptr<llama_ridgefill, decltype(&llama_ridgefill_free)> rf(nullptr, llama_ridgefill_free);
    if (!projector.empty()) {
        rf.reset(llama_ridgefill_init(model.get(), projector.c_str()));
        if (!rf) { throw std::runtime_error("projector load failed"); }
    }
#ifdef RF27_API
    if (!marker_arg.empty()) {
        std::vector<llama_token> m;
        for (int v : int_list(marker_arg)) { m.push_back(v); }
        printf("%d\n", llama_ridgefill_last_marker(tokens.data(), (int32_t) tokens.size(), m.data(), (int32_t) m.size()));
        return 0;
    }
    if (select_share >= 0.0f) {
        if (!rf) { throw std::invalid_argument("--select-share needs --projector"); }
        const int np = n - std::min(tail, n);
        std::vector<llama_pos> rows(std::max(np, 1));
        llama_ridgefill_select_params sp = llama_ridgefill_select_default_params(rf.get());
        sp.share = select_share; sp.digits = select_digits != 0; sp.special_exact = select_special != 0; sp.header_rows = select_header;
        const int32_t k = llama_ridgefill_select_rows_ext(rf.get(), tokens.data(), np, &sp, rows.data());
        if (k < 0) { return 3; }
        write_bytes(output + ".rows.i32", rows.data(), (size_t) k * sizeof(llama_pos));
        fprintf(stderr, "selected %d of %d prefix rows\n", k, np);
        return 0;
    }
#endif
    auto cp = llama_context_default_params();
    cp.n_ctx = ((n + 255) / 256) * 256 * n_seq;
    cp.n_batch = n_batch * n_seq;
    cp.n_ubatch = n_ubatch * n_seq;
    cp.n_seq_max = n_seq;
    cp.kv_unified = true;
    cp.n_rs_seq = n_rs_seq;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) { throw std::runtime_error("context creation failed"); }
    if (rf) {
        if (llama_set_ridgefill(ctx.get(), rf.get()) < 0) { throw std::runtime_error("RidgeFill graphs' compute buffers do not fit"); }
#ifdef RF27_API
        llama_ridgefill_set_alpha(ctx.get(), alpha);
#endif
        for (int seq = 0; seq < n_seq; ++seq) {
        llama_ridgefill_seq_set(ctx.get(), seq, prefix);
#ifdef RF27_API
        const std::string & arg_rows = seq == 0 ? rows_arg : rows1_arg;
        if (mode == "quality" && !arg_rows.empty()) {
            std::vector<llama_pos> rows;
            size_t begin = 0;
            while (begin < arg_rows.size()) {
                size_t end = arg_rows.find(',', begin);
                const int row = std::stoi(arg_rows.substr(begin, end - begin));
                if (row < 0 || row >= prefix || (!rows.empty() && row <= rows.back())) {
                    throw std::invalid_argument("--rows must be sorted unique prefix positions");
                }
                rows.push_back(row);
                if (end == std::string::npos) { break; }
                begin = end + 1;
            }
            if (llama_ridgefill_seq_set_rows(ctx.get(), seq, rows.data(), rows.size()) != 0) {
                throw std::invalid_argument("llama_ridgefill_seq_set_rows refused the rows");
            }
        }
#endif
        }
    }
#ifdef RF27_API
    if (vtail > 0 && !sweep && rf && llama_ridgefill_seq_set_vtail(ctx.get(), 0, true, vtail) != 0) { return 3; } // single pass: refused at decode
#endif
    const std::vector<int> win_layers = int_list(win_arg);
    const std::vector<int> inp_layers = int_list(inp_arg);
    for (int l : inp_layers) {
        llama_set_embeddings_layer_inp(ctx.get(), l, true);
    }
    const int n_embd = llama_model_n_embd(model.get());
    ggml_context_ptr sweep_ctx;
    ggml_backend_buffer_ptr sweep_buf;
    ggml_tensor * h_buf = nullptr;
    std::vector<ggml_tensor *> win_bufs;
    if (warm) {
        // another prompt in the same sequence first, then remove it without zeroing the buffers
        if (rf) { llama_ridgefill_seq_set(ctx.get(), 0, 0); }
        llama_batch wb = llama_batch_init(n_batch, 0, 1);
        for (int start = 0; start < n; start += n_batch) {
            const int count = std::min(n_batch, n - start);
            wb.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                wb.token[i] = tokens[n - 1 - (start + i)]; wb.pos[i] = start + i; wb.n_seq_id[i] = 1; wb.seq_id[i][0] = 0; wb.logits[i] = 0;
            }
            if (llama_decode(ctx.get(), wb) != 0) { llama_batch_free(wb); return 2; }
        }
        llama_batch_free(wb);
        dump_state(ctx.get(), output + ".warm.kv");
        llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1);
        if (rf) { llama_ridgefill_seq_set(ctx.get(), 0, prefix); }
    }
    if (sweep) {
        if (n_seq != 1) { throw std::invalid_argument("--sweep runs one sequence"); }
        ggml_init_params ip = { (1 + win_layers.size()) * ggml_tensor_overhead(), nullptr, true };
        sweep_ctx.reset(ggml_init(ip));
        h_buf = ggml_new_tensor_2d(sweep_ctx.get(), GGML_TYPE_F32, n_embd, n);
        for (size_t k = 0; k < win_layers.size(); ++k) {
            win_bufs.push_back(ggml_new_tensor_2d(sweep_ctx.get(), GGML_TYPE_F32, n_embd, win_n));
        }
        sweep_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(sweep_ctx.get(), ggml_backend_cpu_buffer_type()));
        ggml_backend_buffer_clear(sweep_buf.get(), 0);

        // sweep 1: LOWER over the whole prompt (no outputs)
        llama_ridgefill_part_params pp = llama_ridgefill_part_default_params();
        pp.part = LLAMA_RIDGEFILL_PART_LOWER;
        pp.h_buf = h_buf;
        pp.n_win_layers = win_layers.size();
        pp.win_layers = win_layers.data();
        pp.win_bufs = win_bufs.data();
        pp.win_begin = n - win_n;
        pp.win_n = win_n;
        if (llama_ridgefill_seq_set_part(ctx.get(), 0, &pp) != 0) { return 3; }
        llama_batch lb = llama_batch_init(n_batch, 0, 1);
        for (int start = 0; start < (late_only ? 0 : n); start += n_batch) {
            const int count = std::min(n_batch, n - start);
            lb.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                lb.token[i] = tokens[start + i]; lb.pos[i] = start + i; lb.n_seq_id[i] = 1; lb.seq_id[i][0] = 0;
                lb.logits[i] = filled_logit && start + i == n - 1; // --filled-logit: a LOWER output (refused)
            }
            if (llama_decode(ctx.get(), lb) != 0) { llama_batch_free(lb); fprintf(stderr, "LOWER decode refused\n"); return 2; }
        }
        llama_batch_free(lb);
        filled_logit = false;
        if (lower_twice && llama_ridgefill_seq_set_part(ctx.get(), 0, &pp) != 0) { return 3; }
        if (full_midsweep && llama_ridgefill_seq_set_part(ctx.get(), 0, nullptr) != 0) { return 3; }
#ifdef RF27_API
        if (llama_ridgefill_api_version() != LLAMA_RIDGEFILL_API_VERSION) { throw std::runtime_error("RidgeFill API version mismatch"); }
        if (s1_window > 0) {
            // S1: the speed fill of the late attention layers over [0, P), then the window through the late layers
            auto replay = [&](int b, int e) {
                llama_batch sb = llama_batch_init(n_batch, 0, 1);
                for (int start = b; start < e; start += n_batch) {
                    const int count = std::min(n_batch, e - start);
                    sb.n_tokens = count;
                    for (int i = 0; i < count; ++i) {
                        sb.token[i] = tokens[start + i]; sb.pos[i] = start + i; sb.n_seq_id[i] = 1; sb.seq_id[i][0] = 0; sb.logits[i] = 0;
                    }
                    if (llama_decode(ctx.get(), sb) != 0) { llama_batch_free(sb); return false; }
                }
                llama_batch_free(sb);
                return true;
            };
            pp = llama_ridgefill_part_default_params();
            pp.part = LLAMA_RIDGEFILL_PART_S1_FILL;
            pp.h_buf = h_buf;
            if (llama_ridgefill_seq_set_part(ctx.get(), 0, &pp) != 0) { return 3; }
            if (!replay(0, prefix)) { fprintf(stderr, "S1_FILL decode refused\n"); return 2; }
            if (!s1_late_early) {
                pp = llama_ridgefill_part_default_params();
                pp.part = LLAMA_RIDGEFILL_PART_S1_PROBE;
                pp.h_buf = h_buf;
                pp.s1_begin = s1_begin_arg >= 0 ? s1_begin_arg : n - s1_window;
                pp.s1_end = std::min(n, pp.s1_begin + s1_window);
                if (llama_ridgefill_seq_set_part(ctx.get(), 0, &pp) != 0) { return 3; }
                if (!replay(pp.s1_begin, pp.s1_end)) { fprintf(stderr, "S1_PROBE decode refused\n"); return 2; }
                std::vector<double> mass(prefix);
                if (llama_ridgefill_seq_get_s1_mass(ctx.get(), 0, mass.data(), prefix) != prefix) { throw std::runtime_error("no S1 mass"); }
                write_bytes(output + ".mass_s1.f64", mass.data(), mass.size() * sizeof(double));
                std::vector<int> order(prefix);
                for (int i = 0; i < prefix; ++i) { order[i] = i; }
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return mass[a] > mass[b]; });
                std::vector<llama_pos> rows(order.begin(), order.begin() + std::min(std::max(s1_k, 1), prefix));
                if (s1_special) {
                    std::vector<llama_pos> sp(prefix);
                    llama_ridgefill_select_params sel = llama_ridgefill_select_default_params(rf.get());
                    sel.share = 0.0f; sel.digits = false; sel.special_exact = true; sel.header_rows = 0;
                    const int32_t k = llama_ridgefill_select_rows_ext(rf.get(), tokens.data(), prefix, &sel, sp.data());
                    if (k < 0) { return 3; }
                    rows.insert(rows.end(), sp.begin(), sp.begin() + k);
                }
                std::sort(rows.begin(), rows.end());
                rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
                write_bytes(output + ".rows.i32", rows.data(), rows.size() * sizeof(llama_pos));
                if (llama_ridgefill_seq_set_rows(ctx.get(), 0, rows.data(), rows.size()) != 0) {
                    throw std::invalid_argument("llama_ridgefill_seq_set_rows refused the S1 rows");
                }
            }
        }
        if (vtail > 0 && llama_ridgefill_seq_set_vtail(ctx.get(), 0, true, vtail) != 0) { return 3; }
#endif
        // sweep 2+3: LATE replay over all rows (fill below p_end, exact tail + logits above)
        pp = llama_ridgefill_part_default_params();
        pp.part = LLAMA_RIDGEFILL_PART_LATE;
        pp.h_buf = h_buf;
        if (llama_ridgefill_seq_set_part(ctx.get(), 0, &pp) != 0) { return 3; }
        cross = !sweep_boundary;
    }
    llama_batch batch = llama_batch_init(n_batch * n_seq, 0, 1);
    std::vector<std::vector<float>> logits(n_seq);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    auto name = [&](int seq) { return n_seq == 1 ? output : output + ".s" + std::to_string(seq); };
    for (int start = 0; start < n;) {
        // Also dump stock at the same boundary, to compare prefix states before the tail overwrites them.
        const int boundary = !cross && start < n - std::min(tail, n) ? n - std::min(tail, n) : n;
        const int count = std::min(n_batch, boundary - start);
        batch.n_tokens = count * n_seq;
        for (int seq = 0; seq < n_seq; ++seq) {
            for (int i = 0; i < count; ++i) {
                const int pos = start + i;
                const int j = seq * count + i;
                batch.token[j] = tokens[pos];
                batch.pos[j] = pos;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = seq;
                batch.logits[j] = pos >= n - keep || (filled_logit && pos == 0);
            }
        }
        const int result = llama_decode(ctx.get(), batch);
        if (result != 0) {
            llama_batch_free(batch);
            fprintf(stderr, "llama_decode refused batch: %d\n", result);
            return 2;
        }
        for (int seq = 0; seq < n_seq; ++seq) {
            for (int i = 0; i < count; ++i) {
                if (start + i >= n - keep) {
                    const float * row = llama_get_logits_ith(ctx.get(), seq * count + i);
                    if (!row) { throw std::runtime_error("missing tail logits"); }
                    logits[seq].insert(logits[seq].end(), row, row + vocab);
                }
            }
        }
        if (sweep && start == 0 && !sweep_boundary) {
            dump_state(ctx.get(), output + ".late1.kv"); // the state after the first LATE (replay) batch
        }
        start += count;
        if (!cross && start == n - std::min(tail, n) && start > 0) {
            for (int seq = 0; seq < n_seq; ++seq) {
                dump_state(ctx.get(), name(seq) + ".prefix.kv", seq);
            }
        }
    }
    llama_batch_free(batch);
    if (sweep) {
#ifdef RF27_API
        if (vtail > 0) {
            std::vector<float> sc(4096);
            const int32_t k = llama_ridgefill_seq_get_vtail_scales(ctx.get(), 0, sc.data(), (int32_t) sc.size());
            if (k <= 0) { throw std::runtime_error("vtail on, but no scales were applied"); }
            write_bytes(output + ".vtail.f32", sc.data(), (size_t) k * sizeof(float));
        }
#endif
        llama_ridgefill_seq_set_part(ctx.get(), 0, nullptr);
        write_tensor(output + ".h.f32", h_buf, n);
        for (size_t k = 0; k < win_layers.size(); ++k) {
            write_tensor(output + ".win" + std::to_string(win_layers[k]) + ".f32", win_bufs[k], win_n);
        }
    }
    for (int l : inp_layers) {
        // upstream copies the layer input of every token of the last decode, in batch order: one batch = all rows
        if (n_batch < n) { throw std::invalid_argument("--layer-inp needs the whole prompt in one batch"); }
        const float * e = llama_get_embeddings_layer_inp(ctx.get(), l);
        if (!e) { throw std::runtime_error("no layer inputs"); }
        write_bytes(output + ".inp" + std::to_string(l) + ".f32", e, (size_t) n * n_embd * sizeof(float));
    }
    for (int seq = 0; seq < n_seq; ++seq) {
        if (logits[seq].size() != static_cast<size_t>(keep) * vocab) {
            throw std::runtime_error("incomplete tail logits");
        }
        write_bytes(name(seq) + ".logits", logits[seq].data(), logits[seq].size() * sizeof(float));
        dump_state(ctx.get(), name(seq) + ".kv", seq);
    }
    fprintf(stderr, "dumped n=%d prefix=%d keep=%d vocab=%d mode=%s alpha=%g\n", n, prefix, keep, vocab, mode.c_str(), alpha);
    return 0;
}

int main(int argc, char ** argv) {
    try {
        llama_backend_init();
        const int result = run(argc, argv);
        llama_backend_free();
        return result;
    } catch (const std::exception & error) {
        fprintf(stderr, "rf_dump: %s\n", error.what());
        return 1;
    }
}
