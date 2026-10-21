// Exercise the common selector with the real tiny model/projector and hand-computed positions.
#include "common.h"
#include "llama-ext.h"

#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: test-row-selection TINY_MODEL_GGUF SPLIT3_PROJECTOR_GGUF\n");
        return 2;
    }
    try {
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model || llama_vocab_n_tokens(llama_model_get_vocab(model.get())) != 300) { throw std::runtime_error("requires the tiny 300-token fixture"); }
        std::unique_ptr<llama_ridgefill, decltype(&llama_ridgefill_free)> rf(llama_ridgefill_init(model.get(), argv[2]), llama_ridgefill_free);
        if (!rf || llama_ridgefill_split(rf.get()) != 3) { throw std::runtime_error("requires the tiny split-3 projector"); }
        int checks = 0;
        auto check = [&](bool condition) {
            if (!condition) { throw std::runtime_error("common row selection check failed"); }
            ++checks;
        };
        common_params params;
        params.ridgefill_mode = "quality";
        common_ridgefill_resolve(params, rf.get());
        check(params.ridgefill_share == 0.5f && params.ridgefill_digits == 1);
        const std::vector<llama_token> tokens = {259, 261, 269, 297, 299, 260, 269};
        auto select = [&](const std::vector<llama_token> & ids, const std::string & mode = "") {
            return common_ridgefill_rows(params, rf.get(), ids.data(), ids.size(), mode);
        };
        const std::vector<llama_pos> base = {2, 3, 4, 6};
        check(select(tokens) == base);
        params.ridgefill_digits = 0;
        check(select(tokens) == std::vector<llama_pos>({2, 3, 4}));
        params.ridgefill_share = 0.25f;
        check(select(tokens) == std::vector<llama_pos>({3, 4}));
        params.ridgefill_share = 0.5f;
        check(select({259}).empty());
        check(select({259, 261, 263}) == std::vector<llama_pos>({1, 2}));
        check(select({299, 297, 299, 297, 260}) == std::vector<llama_pos>({0, 1}));
        params.ridgefill_share = 0.25f;
        check(select({269}).empty());
        params.ridgefill_digits = 1;
        check(select({269}) == std::vector<llama_pos>({0}));
        params.ridgefill_share = 0.5f;
        for (float share : {0.0f, 0.2f, 1.0f}) {
            params.ridgefill_qattn_share = share;
            check(select(tokens, "quality+qattn") == base);
        }
        check(select(tokens, "off").empty());
        check(select(tokens, "speed").empty());
        check(common_ridgefill_rows(params, nullptr, nullptr, 0).empty());
        auto refused = [&](auto operation, const std::string & message) {
            bool caught = false;
            try { operation(); }
            catch (const std::exception & error) { caught = std::string(error.what()).find(message) != std::string::npos; }
            check(caught);
        };
        refused([&]() { select({300}); }, "llama_ridgefill_select_rows");
        refused([&]() { select(tokens, "invalid"); }, "invalid mode");
        refused([&]() { common_ridgefill_rows(params, nullptr, tokens.data(), tokens.size()); }, "requires a projector");
        refused([&]() { common_ridgefill_rows(params, rf.get(), tokens.data(), -1); }, "nonnegative prefix");
        params.ridgefill_share = std::numeric_limits<float>::quiet_NaN();
        refused([&]() { select(tokens); }, "--ridgefill-share");
        params.ridgefill_share = 0.5f;
        params.ridgefill_digits = -1;
        refused([&]() { select(tokens); }, "--ridgefill-digits");
        params.ridgefill_digits = 1;
        params.ridgefill_alpha = 1;
        refused([&]() { select(tokens); }, "rho-scaled correction");
        refused([&]() { common_ridgefill_resolve(params, rf.get()); }, "rho-scaled correction");
        check(select(tokens, "off").empty());
        params.ridgefill_alpha = 0;
        params.ridgefill_digits = 0;
        const std::vector<llama_token> conversation = {1, 260, 260, 2, 260, 260, 0};
        check(select(conversation).empty());
        params.ridgefill_special_exact = true;
        check(select(conversation) == std::vector<llama_pos>({0, 3, 6}));
        params.ridgefill_special_exact = false;
        params.ridgefill_header_rows = 2;
        check(select(conversation) == std::vector<llama_pos>({1, 2, 4, 5}));
        params.ridgefill_special_exact = true;
        check(select(conversation) == std::vector<llama_pos>({0, 1, 2, 3, 4, 5, 6}));
        check(select(conversation, "quality+qattn") == select(conversation));
        check(select(conversation, "off").empty());
        refused([&]() { select(conversation, "speed"); }, "require quality");
        refused([&]() { common_ridgefill_rows(params, nullptr, nullptr, 0, "speed"); }, "require quality");
        params.ridgefill_special_exact = false;
        params.ridgefill_header_rows = 0;
        params.ridgefill_digits = 1;
        check(select(tokens) == base);
        params.ridgefill_qattn_window_mode = "turn";
        refused([&]() { select(tokens); }, "last-user-turn");
        check(select(tokens, "off").empty());
        params.ridgefill_qattn_window_mode = "last";
        printf("PASS: %d common class/digit/special/header selector checks\n", checks);
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
