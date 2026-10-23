#include "llama-ridgefill-sweep.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

int main() {
    try {
        int checks = 0;
        auto check = [&](bool ok) {
            if (!ok) { throw std::runtime_error("certificate check " + std::to_string(checks + 1) + " failed"); }
            ++checks;
        };
        constexpr int n = 9, queries = 12;
        ggml_init_params ip = {2*1024*1024, nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        auto * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, queries);
        auto * delta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, queries);
        auto * uncertain = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        auto * mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        for (auto * t : {scores, delta, uncertain, mask}) { ggml_set_input(t); }
        auto * upper = llama_ridgefill_interval_upper(ctx, scores, delta, uncertain, mask);
        ggml_set_output(upper);
        auto * pooled = ggml_pool_1d(ctx, ggml_cont(ctx, ggml_transpose(ctx, upper)), GGML_OP_POOL_MAX, queries, queries, 0);
        ggml_set_output(pooled);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, upper);
        ggml_build_forward_expand(graph, pooled);
        ggml_backend_t backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, 4);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buffer != nullptr);
        std::mt19937 rng(75123);
        std::uniform_real_distribution<float> random(-1, 1);
        std::vector<float> z(n*queries), d(queries), u(n), m(n), observed(n*queries);
        for (int i = 0; i < n; ++i) { u[i] = i < 6; m[i] = i == 7 ? -INFINITY : 0; }
        ggml_backend_tensor_set(uncertain, u.data(), 0, u.size()*sizeof(float));
        ggml_backend_tensor_set(mask, m.data(), 0, m.size()*sizeof(float));
        double largest_slack = 0, largest_shortfall = 0;
        for (int trial = 0; trial < 24; ++trial) {
            for (int q = 0; q < queries; ++q) {
                d[q] = trial == 0 ? 0.0f : trial == 23 ? 1000.0f : 3.0f*std::abs(random(rng));
                for (int i = 0; i < n; ++i) { z[q*n + i] = (trial == 22 ? 100.0f : 8.0f)*random(rng); }
            }
            ggml_backend_tensor_set(scores, z.data(), 0, z.size()*sizeof(float));
            ggml_backend_tensor_set(delta, d.data(), 0, d.size()*sizeof(float));
            check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(upper, observed.data(), 0, observed.size()*sizeof(float));
            for (int q = 0; q < queries; ++q) {
                for (int i = 0; i < n; ++i) {
                    if (!std::isfinite(m[i])) { continue; }
                    // The maximizing corner raises i and lowers every other uncertain key.
                    std::vector<double> corner(n);
                    for (int j = 0; j < n; ++j) {
                        corner[j] = z[q*n + j] + u[j]*d[q]*(j == i ? 1 : -1) + m[j];
                    }
                    const double maximum = *std::max_element(corner.begin(), corner.end());
                    double sum = 0;
                    for (double x : corner) { sum += std::exp(x - maximum); }
                    const double exact_upper = std::exp(corner[i] - maximum)/sum;
                    const double got = observed[q*n + i];
                    check(std::isfinite(got) && got >= 0 && got <= 1 && got + 2e-6 >= exact_upper);
                    largest_slack = std::max(largest_slack, got - exact_upper);
                    largest_shortfall = std::max(largest_shortfall, exact_upper - got);
                    if (trial == 0) { check(std::abs(got - exact_upper) < 3e-6); }
                    for (int sample = 0; sample < 4; ++sample) {
                        for (int j = 0; j < n; ++j) { corner[j] = z[q*n + j] + u[j]*d[q]*random(rng) + m[j]; }
                        const double high = *std::max_element(corner.begin(), corner.end());
                        sum = 0;
                        for (double x : corner) { sum += std::exp(x - high); }
                        check(got + 2e-6 >= std::exp(corner[i] - high)/sum);
                    }
                }
            }
            std::vector<float> maxima(n);
            ggml_backend_tensor_get(pooled, maxima.data(), 0, maxima.size()*sizeof(float));
            for (int i = 0; i < n; ++i) {
                float expected = 0;
                for (int q = 0; q < queries; ++q) { expected = std::max(expected, observed[q*n+i]); }
                check(maxima[i] == expected);
            }
        }
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        llama_ridgefill_certificate_result selection;
        selection.cap = 0.6; selection.tau = 0.3;
        const std::vector<double> bounds = {0.9, 0.4, 0.8, 0.8, 0.1};
        check(llama_ridgefill_certificate_select(bounds, {1}, selection) == std::vector<int32_t>({0, 1, 2}));
        check(selection.flagged == 4 && selection.added == std::vector<int32_t>({0, 2}) && selection.dropped == 1 && selection.capped);
        selection.tau = 1;
        check(llama_ridgefill_certificate_select(bounds, {1}, selection) == std::vector<int32_t>({1}));
        check(selection.flagged == 0 && selection.added.empty() && !selection.capped);
        selection.tau = 0; selection.cap = 1;
        check(llama_ridgefill_certificate_select(bounds, {}, selection).size() == bounds.size());
        auto refused = [&](auto operation) {
            bool caught = false;
            try { operation(); } catch (const std::runtime_error &) { caught = true; }
            check(caught);
        };
        selection.cap = 0;
        refused([&] { llama_ridgefill_certificate_select(bounds, {1}, selection); });
        selection.cap = 1;
        refused([&] { llama_ridgefill_certificate_select(bounds, {2, 1}, selection); });
        refused([&] { llama_ridgefill_certificate_select({NAN}, {}, selection); });
        selection.tau = -1;
        refused([&] { llama_ridgefill_certificate_select(bounds, {}, selection); });
        selection.tau = 0.01; selection.cap = INFINITY;
        refused([&] { llama_ridgefill_certificate_select(bounds, {}, selection); });
        printf("PASS: %d certificate checks; largest FP32 upper slack %.9g, shortfall %.9g\n", checks, largest_slack, largest_shortfall);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
