#include "build-info.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-ext.h"
#include "server-ridgefill-sweep.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

static void require(bool value, const std::string & message) {
    if (!value) { throw std::runtime_error("ridgefill-score: " + message); }
}

struct options {
    std::string model, projector, input, output;
    std::string mode = "quality+qattn", cache = "f16", hidden = "f16", flash = "auto";
    int32_t window_end = -1; // -1: document end; -2: start of scored tokens.
    int32_t scored = 512, tail = 4096, min_prompt = 8192, window = 64;
    int32_t threads = 4, batch = 512, ubatch = 512, ctx = 0, gpu_layers = 0;
    float share = 0.5f, qattn_share = 0.2f, alpha = 0.0f;
    bool digits = true, diagnostics = false;
};

static int32_t integer(const std::string & value) {
    size_t end = 0;
    const auto n = std::stoll(value, &end);
    require(end == value.size() && n >= -1 && n <= INT32_MAX, "invalid integer: " + value);
    return int32_t(n);
}

static float fraction(const std::string & value) {
    size_t end = 0;
    const float x = std::stof(value, &end);
    require(end == value.size() && std::isfinite(x) && x >= 0 && x <= 1, "fraction must be in [0,1]: " + value);
    return x;
}

static ggml_type tensor_type(const std::string & value) {
    if (value == "f32") { return GGML_TYPE_F32; }
    if (value == "f16") { return GGML_TYPE_F16; }
    if (value == "bf16") { return GGML_TYPE_BF16; }
    throw std::runtime_error("ridgefill-score: type must be f32, f16 or bf16");
}

static options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--diagnostics") { o.diagnostics = true; continue; }
        require(i + 1 < argc, "missing value for " + key);
        const std::string value = argv[++i];
        if      (key == "--model" || key == "-m") { o.model = value; }
        else if (key == "--projector" || key == "--ridgefill") { o.projector = value; }
        else if (key == "--input") { o.input = value; }
        else if (key == "--output") { o.output = value; }
        else if (key == "--mode") { o.mode = value; }
        else if (key == "--scored") { o.scored = integer(value); }
        else if (key == "--tail") { o.tail = integer(value); }
        else if (key == "--min-prompt") { o.min_prompt = integer(value); }
        else if (key == "--qattn-window") { o.window = integer(value); }
        else if (key == "--qattn-window-end" || key == "--qattn-end") {
            if (value == "document") { o.window_end = -1; }
            else if (value == "known") { o.window_end = -2; }
            else {
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), o.window_end);
                require(parsed.ec == std::errc() && parsed.ptr == value.data() + value.size() && o.window_end > 0,
                        "--qattn-window-end must be a positive row index, document or known");
            }
        }
        else if (key == "--threads" || key == "-t") { o.threads = integer(value); }
        else if (key == "--batch" || key == "-b") { o.batch = integer(value); }
        else if (key == "--ubatch" || key == "-ub") { o.ubatch = integer(value); }
        else if (key == "--ctx-size" || key == "-c") { o.ctx = integer(value); }
        else if (key == "--gpu-layers" || key == "-ngl") { o.gpu_layers = integer(value); }
        else if (key == "--cache-type") { o.cache = value; }
        else if (key == "--h-type") { o.hidden = value; }
        else if (key == "--flash-attn" || key == "-fa") { o.flash = value; }
        else if (key == "--share") { o.share = fraction(value); }
        else if (key == "--qattn-share") { o.qattn_share = fraction(value); }
        else if (key == "--alpha") { o.alpha = fraction(value); }
        else if (key == "--digits") { require(value == "on" || value == "off", "digits must be on or off"); o.digits = value == "on"; }
        else { throw std::runtime_error("ridgefill-score: unknown option " + key); }
    }
    require(!o.model.empty() && !o.input.empty() && !o.output.empty(), "--model, --input and --output are required");
    require(o.mode == "off" || o.mode == "speed" || o.mode == "quality" || o.mode == "quality+qattn", "invalid mode");
    require(o.mode == "off" || !o.projector.empty(), "active modes require --projector");
    require(o.share > 0, "class share must be in (0,1]");
    require((o.mode != "quality" && o.mode != "quality+qattn") || o.alpha == 0,
            "quality modes with nonzero alpha require the rho-scaled correction; use --alpha 0");
    require(o.scored > 0 && o.tail > 0 && o.min_prompt >= 0 && o.window > 0 && o.threads > 0 && o.batch > 0 && o.ubatch > 0 && o.ctx >= 0, "invalid count");
    require(o.mode == "off" || o.tail > o.scored, "tail must exceed scored so the first prediction row is exact");
    require(o.flash == "off" || o.flash == "on" || o.flash == "auto", "invalid flash-attn mode");
    tensor_type(o.cache); tensor_type(o.hidden);
    return o;
}

struct document {
    std::string name;
    std::vector<llama_token> ids;
};

static std::vector<document> read_documents(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file), "cannot open input " + path);
    const auto size = file.tellg();
    require(size > 0 && size <= 128*1024*1024, "input must be nonempty and at most 128 MiB");
    std::string text(size_t(size), '\0');
    file.seekg(0); file.read(&text[0], size);
    require(bool(file), "cannot read input");
    std::vector<json> records;
    auto whole = json::parse(text, nullptr, false);
    if (!whole.is_discarded()) {
        if (whole.is_array() && !whole.empty() && whole.front().is_object()) {
            for (const auto & row : whole) { records.push_back(row); }
        } else { records.push_back(std::move(whole)); }
    } else {
        std::istringstream lines(text);
        std::string line;
        while (std::getline(lines, line)) {
            if (line.find_first_not_of(" \t\r") != std::string::npos) { records.push_back(json::parse(line)); }
        }
    }
    std::vector<document> docs;
    std::set<std::string> names;
    for (const auto & record : records) {
        const auto & ids = record.is_object() ? record.at("ids") : record;
        require(ids.is_array() && !ids.empty() && ids.size() <= 1048576, "each document needs 1..1048576 token IDs");
        document doc;
        doc.name = record.is_object() ? record.value("name", std::to_string(docs.size())) : std::to_string(docs.size());
        require(names.insert(doc.name).second, "duplicate document name: " + doc.name);
        for (const auto & id : ids) {
            require(id.is_number_integer() && id >= 0 && id <= INT32_MAX, "token IDs must be nonnegative int32 values");
            doc.ids.push_back(id.get<llama_token>());
        }
        docs.push_back(std::move(doc));
    }
    require(!docs.empty(), "no documents");
    return docs;
}

static json score(llama_context * ctx, const llama_model * model, const llama_ridgefill * rf, const document & doc, const options & o) {
    const auto start = std::chrono::steady_clock::now();
    const int32_t n = doc.ids.size(), begin = n - o.scored - 1, end = n - 1;
    const int32_t prefix = o.mode != "off" && n >= o.min_prompt ? std::max(0, n - o.tail) : 0;
    const bool qattn = prefix > 0 && o.mode == "quality+qattn" && o.qattn_share > 0;
    const int32_t window_end = o.window_end == -2 ? n - o.scored : (o.window_end == -1 ? n : o.window_end);
    require(!qattn || window_end >= o.window, "--qattn-window-end is shorter than --qattn-window");
    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_ridgefill_seq_set(ctx, 0, prefix);
    common_params selection;
    selection.ridgefill_mode = o.mode; selection.ridgefill_share = o.share;
    selection.ridgefill_digits = o.digits; selection.ridgefill_alpha = o.alpha;
    auto rows = common_ridgefill_rows(selection, rf, doc.ids.data(), prefix, o.mode);
    require(llama_ridgefill_seq_set_rows(ctx, 0, rows.data(), rows.size()) == 0, "exact row setup failed");
    std::vector<double> logprobs, logsumexp;
    std::vector<int32_t> positions;
    auto collect = [&](int32_t first, int32_t count) {
        for (int32_t i = 0; i < count; ++i) {
            const int32_t pos = first + i;
            require(pos == begin + int32_t(logprobs.size()) && pos < end, "output row order/count differs from scoring range");
            const float * logits = llama_get_logits_ith(ctx, i - count);
            require(logits != nullptr, "missing logits");
            double maximum = -std::numeric_limits<double>::infinity();
            for (int32_t v = 0; v < vocab; ++v) {
                if (!std::isfinite(logits[v])) {
                    throw std::runtime_error("ridgefill-score: non-finite full-vocabulary logits at row " + std::to_string(pos));
                }
                maximum = std::max(maximum, double(logits[v]));
            }
            double sum = 0;
            for (int32_t v = 0; v < vocab; ++v) { sum += std::exp(double(logits[v]) - maximum); }
            const double lse = maximum + std::log(sum);
            logsumexp.push_back(lse);
            logprobs.push_back(double(logits[doc.ids[pos + 1]]) - lse);
            positions.push_back(pos + 1);
        }
    };
    server_ridgefill_sweep sweep;
    if (qattn) {
        server_ridgefill_sweep_params p;
        p.n_embd = llama_model_n_embd(model); p.n_tokens = n; p.n_prefix = prefix; p.n_batch = o.batch;
        p.split = llama_ridgefill_split(rf); p.qattn_share = o.qattn_share; p.qattn_window = o.window;
        p.qattn_end = window_end; p.logits_begin = begin; p.logits_end = end; p.type = tensor_type(o.hidden);
        try {
            while (true) {
                const auto result = server_ridgefill_sweep_step(sweep, ctx, 0, doc.ids.data(), rows, p);
                collect(sweep.first_output_position(), sweep.last_outputs());
                if (result == server_ridgefill_sweep_result::complete) { break; }
                require(result == server_ridgefill_sweep_result::pending, "unexpected sweep fallback/deferred output");
            }
        } catch (...) { sweep.cancel(ctx, 0); throw; }
        rows = sweep.get_rows();
    } else {
        for (int32_t first = 0; first < n;) {
            const int32_t boundary = first < prefix ? prefix : n;
            const int32_t count = std::min({o.batch, int32_t(llama_n_ubatch(ctx)), boundary - first});
            llama_batch batch = llama_batch_init(count, 0, 1);
            batch.n_tokens = count;
            int32_t outputs = 0, first_output = -1;
            for (int32_t i = 0; i < count; ++i) {
                batch.token[i] = doc.ids[first + i]; batch.pos[i] = first + i;
                batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                batch.logits[i] = first + i >= begin && first + i < end;
                if (batch.logits[i]) { if (outputs == 0) { first_output = first + i; } ++outputs; }
            }
            const int rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
            require(rc == 0, "decode failed with code " + std::to_string(rc));
            collect(first_output, outputs);
            first += count;
        }
    }
    require(logprobs.size() == size_t(o.scored), "did not score exactly K targets");
    double mean = 0;
    for (double lp : logprobs) { mean -= lp / o.scored; }
    json result = {{"name", doc.name}, {"mode", o.mode}, {"effective_mode", prefix == 0 ? "off" : o.mode},
        {"n", n}, {"scored", o.scored}, {"prefix", prefix}, {"filled", prefix}, {"nll", mean},
        {"logprobs", logprobs}, {"logsumexp", logsumexp}, {"target_positions", positions},
        {"logits_begin", begin}, {"logits_end", end}, {"exact_rows", rows},
        {"qattn_window", qattn ? json::array({window_end - o.window, window_end}) : json(nullptr)},
        {"selector_excludes_targets", !qattn || window_end <= n - o.scored},
        {"parameters", {{"tail", o.tail}, {"share", o.share}, {"digits", o.digits}, {"qattn_share", o.qattn_share},
            {"alpha", o.alpha}, {"cache_type", o.cache}, {"h_type", o.hidden}, {"flash_attn", o.flash}}},
        {"prefills", 1}, {"wall_s", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()},
        {"build", {{"commit", llama_commit()}, {"build", llama_build_number()}}}};
    if (o.diagnostics && qattn) { result["question_mass"] = sweep.get_mass(); }
    return result;
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--version") { llama_print_build_info("llama-ridgefill-score"); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "llama-ridgefill-score --model MODEL --projector GGUF --input JSON/JSONL --output JSONL [--mode off|speed|quality|quality+qattn]\n"
                "Defaults: --scored 512 --tail 4096 --min-prompt 8192 --share 0.5 --digits on --qattn-share 0.2 --qattn-window 64 --alpha 0\n"
                "--qattn-window-end END uses [END-window,END), with an exclusive zero-based endpoint.\n"
                "Default: document (N, server semantics). For NLL use known (N-K), or an explicit row index.\n"
                "--qattn-end is an alias. NLL with endpoint N reads scored targets; the G3 wrapper uses N-K.\n"
                "Runtime: --gpu-layers 0 --threads 4 --batch 512 --ubatch 512 --ctx-size 0 --cache-type f16 --h-type f16 --flash-attn auto\n"
                "JSON: token ID array, {name,ids}, array of documents, or one document per JSONL line. No tokenization.\n";
            return 0;
        }
        const options o = parse(argc, argv);
        const auto docs = read_documents(o.input);
        int32_t max_n = 0;
        for (const auto & doc : docs) {
            require(doc.ids.size() > size_t(o.scored), "document must contain more than K tokens");
            require(o.window_end < 0 || size_t(o.window_end) <= doc.ids.size(), "--qattn-window-end exceeds document length");
            max_n = std::max(max_n, int32_t(doc.ids.size()));
        }
        require(o.ctx == 0 || o.ctx >= max_n, "ctx-size is smaller than a document");
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = o.gpu_layers;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(o.model.c_str(), mp), llama_model_free);
        require(bool(model), "model load failed");
        const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (const auto & doc : docs) { for (auto id : doc.ids) { require(id < vocab, "token ID outside model vocabulary"); } }
        std::unique_ptr<llama_ridgefill, decltype(&llama_ridgefill_free)> rf(
                o.mode == "off" ? nullptr : llama_ridgefill_init(model.get(), o.projector.c_str()), llama_ridgefill_free);
        require(o.mode == "off" || bool(rf), "projector load failed");
        require(!rf || o.tail >= llama_ridgefill_tail_min(rf.get()), "tail is below this projector's tail_min");
        auto cp = llama_context_default_params();
        cp.n_ctx = o.ctx ? o.ctx : max_n; cp.n_batch = o.batch; cp.n_ubatch = o.ubatch; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = o.threads; cp.type_k = cp.type_v = tensor_type(o.cache);
        cp.flash_attn_type = o.flash == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED :
                (o.flash == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO);
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "context creation failed");
        llama_set_ridgefill(ctx.get(), rf.get());
        require(!rf || llama_ridgefill_set_alpha(ctx.get(), o.alpha) == 0, "alpha was refused");
        std::ofstream output(o.output);
        require(bool(output), "cannot open output " + o.output);
        for (const auto & doc : docs) {
            const json result = score(ctx.get(), model.get(), rf.get(), doc, o);
            output << result.dump() << '\n'; output.flush();
            require(bool(output), "cannot write output");
            std::cerr << "ridgefill-score: " << doc.name << " " << o.mode << " NLL=" << result["nll"] << " K=" << o.scored << '\n';
        }
        ctx.reset(); rf.reset(); model.reset(); llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
