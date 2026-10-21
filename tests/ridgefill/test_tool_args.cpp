// Test the shared option parser and prompt boundary without an engine or projector implementation.
#include "common.h"

#include <cstdio>
#include <stdexcept>
#include <string>

int main() {
    int checked = 0;
    auto check = [&](bool condition) {
        if (!condition) { throw std::runtime_error("RidgeFill argument check failed"); }
        ++checked;
    };
    common_params params;
    params.ridgefill_path = "projector.gguf";
    params.ridgefill_tail = 4096;
    check(common_ridgefill_p_end(params, 16384) == 12288);
    check(common_ridgefill_p_end(params, 700) == 0);
    check(common_ridgefill_p_end(params, 16384, 16384) == 0);
    common_ridgefill_set_option(params, "--ridgefill-mode", "off");
    check(common_ridgefill_p_end(params, 16384) == 0);
    check(common_ridgefill_p_end(params, 16384, 4096, "quality") == 12288);
    common_ridgefill_set_option(params, "--ridgefill-mode", "quality");
    common_ridgefill_set_option(params, "--ridgefill-share", "0.5");
    common_ridgefill_set_option(params, "--ridgefill-alpha", "0");
    common_ridgefill_set_option(params, "--ridgefill-digits", "off");
    common_ridgefill_set_option(params, "--ridgefill-qattn-share", "0");
    common_ridgefill_set_option(params, "--ridgefill-qattn-window", "32");
    common_ridgefill_set_option(params, "--ridgefill-qattn-blind", "on");
    check(params.ridgefill_mode == "quality" && params.ridgefill_share == 0.5f && params.ridgefill_alpha == 0);
    check(params.ridgefill_digits == 0 && params.ridgefill_qattn_share == 0 && params.ridgefill_qattn_window == 32 && params.ridgefill_qattn_blind);
    for (const auto & bad : {std::pair<std::string, std::string>{"--ridgefill-mode", "unknown"},
            {"--ridgefill-alpha", "nan"}, {"--ridgefill-alpha", "1.1"}, {"--ridgefill-share", "0"},
            {"--ridgefill-share", "0.5garbage"}, {"--ridgefill-qattn-share", "-1"},
            {"--ridgefill-qattn-window", "0"}, {"--ridgefill-qattn-window", "32garbage"},
            {"--ridgefill-digits", "maybe"}, {"--ridgefill-tail", "-1"}}) {
        bool refused = false;
        try {
            common_ridgefill_set_option(params, bad.first, bad.second);
        } catch (const std::invalid_argument & error) {
            refused = std::string(error.what()).find(bad.first) != std::string::npos;
        }
        check(refused);
    }
    params.ridgefill_tail = 0;
    bool refused = false;
    try { common_ridgefill_p_end(params, 16384); }
    catch (const std::invalid_argument &) { refused = true; }
    check(refused);
    printf("PASS: %d RidgeFill argument/boundary checks\n", checked);
    return 0;
}
