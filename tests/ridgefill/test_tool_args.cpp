// Test the shared option parser and prompt boundary without an engine or projector implementation.
#include "common.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

int main() {
    int checked = 0;
    auto check = [&](bool condition) {
        if (!condition) { throw std::runtime_error("RidgeFill argument check failed"); }
        ++checked;
    };
    common_params retired;
    for (const auto & option : {"--ridgefill-certificate", "--ridgefill-certificate-tau", "--ridgefill-certificate-cap"}) {
        bool caught = false;
        try { common_ridgefill_set_option(retired, option, "off"); }
        catch (const std::invalid_argument &) { caught = true; }
        check(caught);
    }
    common_params params;
    check(!params.ridgefill_special_exact && params.ridgefill_header_rows == 0 && params.ridgefill_qattn_window_mode == "last" && params.ridgefill_qattn_turn_cap == 256);
    check(params.ridgefill_s1_window == 0 && params.ridgefill_s1_factor == 1.0f && !params.ridgefill_vtail && params.ridgefill_vtail_rows == 512);
    common_ridgefill_set_option(params, "--ridgefill-s1-window", "256");
    common_ridgefill_set_option(params, "--ridgefill-s1-factor", "2.5");
    common_ridgefill_set_option(params, "--ridgefill-vtail", "on");
    common_ridgefill_set_option(params, "--ridgefill-vtail-rows", "32");
    check(params.ridgefill_s1_window == 256 && params.ridgefill_s1_factor == 2.5f && params.ridgefill_vtail && params.ridgefill_vtail_rows == 32);
    common_ridgefill_set_option(params, "--ridgefill-s1-factor", "0.1");
    check(std::nearbyint(params.ridgefill_s1_factor * 25) == 2);
    common_ridgefill_set_option(params, "--ridgefill-s1-window", "0");
    common_ridgefill_set_option(params, "--ridgefill-s1-factor", "1");
    common_ridgefill_set_option(params, "--ridgefill-vtail", "off");
    common_ridgefill_set_option(params, "--ridgefill-vtail-rows", "512");
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
    common_ridgefill_set_option(params, "--ridgefill-special-exact", "on");
    common_ridgefill_set_option(params, "--ridgefill-header-rows", "3");
    common_ridgefill_set_option(params, "--ridgefill-qattn-window-mode", "turn");
    check(params.ridgefill_special_exact && params.ridgefill_header_rows == 3 && params.ridgefill_qattn_window_mode == "turn");
    bool turn_refused = false;
    try { common_ridgefill_validate_options(params); }
    catch (const std::invalid_argument & error) { turn_refused = std::string(error.what()).find("quality+qattn") != std::string::npos; }
    check(turn_refused);
    params.ridgefill_mode = "quality+qattn";
    turn_refused = false;
    try { common_ridgefill_validate_options(params); }
    catch (const std::invalid_argument & error) { turn_refused = std::string(error.what()).find("blind") != std::string::npos; }
    check(turn_refused);
    params.ridgefill_qattn_blind = false;
    common_ridgefill_set_option(params, "--qattn-turn-cap", "32");
    common_ridgefill_validate_options(params);
    check(params.ridgefill_qattn_turn_cap == 32);
    common_ridgefill_set_option(params, "--ridgefill-qattn-turn-cap", "256");
    check(params.ridgefill_qattn_turn_cap == 256);
    common_ridgefill_set_option(params, "--ridgefill-qattn-window-mode", "last");
    common_ridgefill_set_option(params, "--ridgefill-header-rows", "0");
    check(params.ridgefill_header_rows == 0);
    common_ridgefill_set_option(params, "--ridgefill-share", "0");
    check(params.ridgefill_share == 0);
    for (const auto & bad : {std::pair<std::string, std::string>{"--ridgefill-mode", "unknown"},
            {"--ridgefill-s1-window", "-1"}, {"--ridgefill-s1-window", "2.5"}, {"--ridgefill-s1-window", "2147483648"},
            {"--ridgefill-s1-factor", "-1"}, {"--ridgefill-s1-factor", "nan"}, {"--ridgefill-s1-factor", "inf"},
            {"--ridgefill-s1-factor", "1garbage"}, {"--ridgefill-vtail", "maybe"}, {"--ridgefill-vtail-rows", "0"},
            {"--ridgefill-vtail-rows", "9999999999999"}, {"--ridgefill-vtail-rows", "1.5"},
            {"--ridgefill-alpha", "nan"}, {"--ridgefill-alpha", "1.1"}, {"--ridgefill-share", "-0.1"},
            {"--ridgefill-share", "0.5garbage"}, {"--ridgefill-qattn-share", "-1"},
            {"--qattn-turn-cap", "0"}, {"--ridgefill-qattn-turn-cap", "2garbage"}, {"--ridgefill-qattn-turn-cap", "9999999999999"},
            {"--ridgefill-qattn-window", "0"}, {"--ridgefill-qattn-window", "32garbage"},
            {"--ridgefill-digits", "maybe"}, {"--ridgefill-tail", "-1"},
            {"--ridgefill-special-exact", "maybe"}, {"--ridgefill-header-rows", "-1"},
            {"--ridgefill-header-rows", "2garbage"}, {"--ridgefill-qattn-window-mode", "unknown"}}) {
        bool refused = false;
        try {
            common_ridgefill_set_option(params, bad.first, bad.second);
        } catch (const std::invalid_argument & error) {
            refused = std::string(error.what()).find(bad.first) != std::string::npos;
        }
        check(refused);
    }
    check(!params.ridgefill_lexical && params.ridgefill_lexical_window == 64 && params.ridgefill_lexical_min_words == 4 &&
            params.ridgefill_lexical_span_rows == 80 && params.ridgefill_lexical_max_occurrences == 2);
    common_ridgefill_set_option(params, "--ridgefill-lexical-window", "128");
    check(params.ridgefill_lexical_window == 128);
    for (const auto & bad : {"0", "-1", "1.5", "4097", "9999999999999", "2bad"}) {
        bool caught = false;
        try { common_ridgefill_set_option(params, "--ridgefill-lexical-window", bad); }
        catch (const std::invalid_argument &) { caught = true; }
        check(caught);
    }
    common_ridgefill_set_option(params, "--ridgefill-lexical-min-words", "128");
    check(params.ridgefill_lexical_min_words == 128);
    for (const auto & bad : {"0", "-1", "1.5", "4097", "9999999999999", "2bad"}) {
        bool caught = false;
        try { common_ridgefill_set_option(params, "--ridgefill-lexical-min-words", bad); }
        catch (const std::invalid_argument &) { caught = true; }
        check(caught);
    }
    common_ridgefill_set_option(params, "--ridgefill-lexical-span-rows", "128");
    check(params.ridgefill_lexical_span_rows == 128);
    for (const auto & bad : {"0", "-1", "1.5", "4097", "9999999999999", "2bad"}) {
        bool caught = false;
        try { common_ridgefill_set_option(params, "--ridgefill-lexical-span-rows", bad); }
        catch (const std::invalid_argument &) { caught = true; }
        check(caught);
    }
    common_ridgefill_set_option(params, "--ridgefill-lexical-max-occurrences", "128");
    check(params.ridgefill_lexical_max_occurrences == 128);
    for (const auto & bad : {"0", "-1", "1.5", "4097", "9999999999999", "2bad"}) {
        bool caught = false;
        try { common_ridgefill_set_option(params, "--ridgefill-lexical-max-occurrences", bad); }
        catch (const std::invalid_argument &) { caught = true; }
        check(caught);
    }
    bool minimum_refused = false;
    try { common_ridgefill_set_option(params, "--ridgefill-lexical-min-words", "1"); }
    catch (const std::invalid_argument &) { minimum_refused = true; }
    check(minimum_refused);
    params.ridgefill_tail = 0;
    bool refused = false;
    try { common_ridgefill_p_end(params, 16384); }
    catch (const std::invalid_argument &) { refused = true; }
    check(refused);
    printf("PASS: %d RidgeFill argument/boundary checks\n", checked);
    return 0;
}
