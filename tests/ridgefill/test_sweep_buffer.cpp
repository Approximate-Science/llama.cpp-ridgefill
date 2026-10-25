#include "llama-ridgefill-sweep.h"

#include <cstdio>
#include <limits>
#include <stdexcept>

static void require(bool ok) {
    if (!ok) { throw std::runtime_error("unexpected sweep buffer size"); }
}

int main() {
    try {
        const size_t gib = size_t(1) << 30;
        const auto bytes = [](int32_t width, int32_t rows, ggml_type type, size_t limit = LLAMA_RIDGEFILL_SWEEP_MAX_BYTES) {
            return llama_ridgefill_sweep_buffer::required_bytes(width, rows, type, limit);
        };
        require(bytes(5120, 65536, GGML_TYPE_F16) == 640*size_t(1 << 20));
        require(bytes(5120, 131072, GGML_TYPE_F16) == 1280*size_t(1 << 20));
        require(bytes(5120, 131072, GGML_TYPE_BF16) == 1280*size_t(1 << 20));
        require(bytes(5120, 65536, GGML_TYPE_F32) == 1280*size_t(1 << 20));
        require(bytes(5120, 209715, GGML_TYPE_F16) == size_t(5120)*209715*2);
        int refusals = 0;
        const auto refuses = [&](int32_t width, int32_t rows, ggml_type type, size_t limit) {
            try { bytes(width, rows, type, limit); }
            catch (const std::runtime_error &) { ++refusals; return; }
            throw std::runtime_error("invalid sweep buffer dimensions accepted");
        };
        refuses(5120, 209716, GGML_TYPE_F16, 2*gib);
        refuses(5120, 104858, GGML_TYPE_F32, 2*gib);
        refuses(5120, 104858, GGML_TYPE_F16, gib);
        refuses(std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max(), GGML_TYPE_F32, 2*gib);
        refuses(0, 1, GGML_TYPE_F16, 2*gib);
        refuses(1, -1, GGML_TYPE_F16, 2*gib);
        refuses(1, 1, GGML_TYPE_Q8_0, 2*gib);
        require(refusals == 7);
        std::puts("PASS: long-context sweep sizing and seven checked refusals");
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
