// Run the production selector without model weights or inference.
#include "common.h"
#include "json.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

using json = nlohmann::json;

static json encode(const common_ridgefill_lexical_result & net) {
    common_ridgefill_report report;
    report.lexical_enabled = true;
    report.lexical = net;
    return json::parse(report.to_json().dump()).at("lexical");
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: lexical-rows MODEL_GGUF INPUT_JSONL OUTPUT_JSONL\n";
        return 2;
    }
    try {
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.vocab_only = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) { throw std::runtime_error("vocabulary load failed"); }
        const auto * vocab = llama_model_get_vocab(model.get());
        std::ifstream input(argv[2]);
        std::ofstream output(argv[3]);
        if (!input || !output) { throw std::runtime_error("cannot open JSONL input/output"); }
        std::string line;
        int count = 0;
        while (std::getline(input, line)) {
            const auto source = json::parse(line);
            auto ids = source.at("ids").get<std::vector<llama_token>>();
            const int32_t n = ids.size(), p = source.at("n_prefix"), e = source.value("selector_end", n);
            if (source.contains("pieces_hex")) {
                const char * hex = "0123456789abcdef";
                const int mask = LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_UNUSED;
                for (int32_t i = 0; i < e; ++i) {
                    std::string piece;
                    for (unsigned char c : common_token_to_piece(vocab, ids[i])) { piece += hex[c >> 4]; piece += hex[c & 15]; }
                    if (piece != source.at("pieces_hex").at(i) || bool(llama_vocab_get_attr(vocab, ids[i]) & mask) != source.at("excluded").at(i).get<bool>()) {
                        throw std::runtime_error("GGUF piece/barrier differs from frozen fixture " + source.at("id").get<std::string>());
                    }
                }
            }
            common_ridgefill_lexical_params params;
            if (source.contains("params")) {
                const auto & knobs = source.at("params");
                params.window_tokens = knobs.value("window_tokens", params.window_tokens);
                params.minimum_words = knobs.value("minimum_words", params.minimum_words);
                params.span_rows = knobs.value("span_rows", params.span_rows);
                params.maximum_occurrences = knobs.value("maximum_occurrences", params.maximum_occurrences);
            }
            const auto observed = encode(common_ridgefill_lexical_rows(vocab, ids.data(), n, p, e, params));
            // Unavailable target IDs must not affect words, boundaries or validation.
            std::fill(ids.begin() + e, ids.end(), INT32_MAX);
            if (encode(common_ridgefill_lexical_rows(vocab, ids.data(), n, p, e, params)) != observed) {
                throw std::runtime_error("selector consulted target suffix");
            }
            output << json {{"id", source.at("id")}, {"lexical", observed}}.dump() << '\n';
            ++count;
        }
        if (!input.eof() || !output || count == 0) { throw std::runtime_error("empty or failed JSONL run"); }
        std::cerr << "PASS: " << count << " production lexical selections and target-suffix invariance checks\n";
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
