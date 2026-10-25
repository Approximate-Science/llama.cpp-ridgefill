#include "../../tools/ridgefill/score-logprobs.h"

#include <filesystem>
#include <iostream>
#include <iterator>

int main(int argc, char ** argv) {
    try {
        if (argc != 2) { throw std::runtime_error("usage: test-score-kl OUTPUT_DIRECTORY"); }
        const std::filesystem::path directory(argv[1]);
        std::filesystem::create_directories(directory);
        int checks = 0;
        auto check = [&](bool ok) {
            if (!ok) { throw std::runtime_error("score KL check " + std::to_string(checks + 1) + " failed"); }
            ++checks;
        };
        auto refused = [&](auto operation, const std::string & message) {
            bool caught = false;
            try { operation(); }
            catch (const std::runtime_error & error) { caught = std::string(error.what()).find(message) != std::string::npos; }
            check(caught);
        };
        const std::vector<float> p = {float(std::log(0.5)), float(std::log(0.25)), float(std::log(0.25))};
        const std::vector<float> q = {p[1], p[0], p[2]};
        ridgefill_kl_statistics statistics;
        statistics.add(p, q, 0);
        check(std::abs(statistics.kl[0] - 0.25*std::log(2.0)) < 1e-8);
        check(!statistics.top1[0]);
        check(std::abs(statistics.dp_target_sum + 0.25) < 1e-8);
        statistics.add(p, p, 1);
        check(statistics.kl[1] == 0 && statistics.top1[1]);
        check(statistics.top1_agree() == 0.5);
        check(statistics.mean() == statistics.kl[0]/2);
        check(statistics.percentile(0.5) == statistics.kl[0]/2);
        check(statistics.percentile(0.99) == 0.99*statistics.kl[0]);
        check(statistics.percentile(1) == statistics.kl[0]);
        check(statistics.percentile(0) == 0);

        const std::vector<float> a = {float(std::log(0.75)), float(std::log(0.25))};
        const std::vector<float> b = {float(std::log(0.5)), float(std::log(0.5))};
        ridgefill_kl_statistics asymmetric, reverse;
        asymmetric.add(a, b, 1);
        reverse.add(b, a, 1);
        check(std::abs(asymmetric.kl[0] - (0.75*std::log(1.5) + 0.25*std::log(0.5))) < 2e-8);
        check(std::abs(reverse.kl[0] - (0.5*std::log(2.0/3.0) + 0.5*std::log(2.0))) < 2e-8);
        check(asymmetric.kl[0] != reverse.kl[0] && asymmetric.top1[0]);
        const std::vector<float> tail_p = {float(std::log(1 - 1e-8)), float(std::log(1e-8))};
        const std::vector<float> tail_q = {float(std::log(1 - 2e-8)), float(std::log(2e-8))};
        ridgefill_kl_statistics tail;
        tail.add(tail_p, tail_q, 1);
        check(std::abs(tail.kl[0] - (1e-8 - 1e-8*std::log(2.0))) < 1e-14);

        const std::vector<int32_t> targets = {1, 2};
        const auto path = (directory / "roundtrip.rflp").string();
        {
            ridgefill_logprob_writer writer(path, 3, targets);
            writer.write(p); writer.write(q); writer.finish();
        }
        std::ifstream raw(path, std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(raw)), {});
        const std::vector<uint8_t> header = {'R', 'F', 'L', 'P', 1, 0, 0, 0, 32, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0};
        check(bytes.size() == 52 && std::equal(header.begin(), header.end(), bytes.begin()));
        check(std::vector<uint8_t>(bytes.begin() + 28, bytes.begin() + 32) == std::vector<uint8_t>({0x18, 0x72, 0x31, 0xbf}));
        ridgefill_logprob_reader reader(path, 3, targets);
        std::vector<float> row;
        reader.read(row); check(row == p);
        reader.read(row); check(row == q);
        refused([&]() { ridgefill_logprob_reader bad(path, 4, targets); }, "n_vocab mismatch");
        refused([&]() { ridgefill_logprob_reader bad(path, 3, {1}); }, "n_scored mismatch");
        refused([&]() { ridgefill_logprob_reader bad(path, 3, {1, 0}); }, "target ids mismatch at scored position 1");

        const auto damaged = (directory / "damaged.rflp").string();
        auto corrupt = [&](size_t offset, uint8_t value, const std::string & message) {
            auto changed = bytes;
            changed[offset] = value;
            std::ofstream file(damaged, std::ios::binary);
            file.write(reinterpret_cast<const char *>(changed.data()), changed.size()); file.close();
            refused([&]() { ridgefill_logprob_reader bad(damaged, 3, targets); bad.read(row); }, message);
        };
        corrupt(0, 'X', "magic");
        corrupt(4, 2, "version");
        corrupt(8, 16, "dtype");
        corrupt(31, 0x7f, "invalid log-probability");
        {
            std::ofstream file(damaged, std::ios::binary);
            file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size() - 1);
        }
        refused([&]() { ridgefill_logprob_reader bad(damaged, 3, targets); }, "file size mismatch");
        {
            std::ofstream file(damaged, std::ios::binary);
            file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size()); file.put(0);
        }
        refused([&]() { ridgefill_logprob_reader bad(damaged, 3, targets); }, "file size mismatch");

        const auto atomic = (directory / "atomic.rflp").string();
        { std::ofstream file(atomic); file << "previous output"; }
        {
            ridgefill_logprob_writer writer(atomic, 3, targets);
            writer.write(p);
            refused([&]() { writer.finish(); }, "n_scored mismatch");
        }
        std::ifstream previous(atomic);
        check(std::string((std::istreambuf_iterator<char>(previous)), {}) == "previous output");
        check(!std::filesystem::exists(atomic + ".tmp"));
        std::cout << "PASS: " << checks << " score KL math, float32 file, mismatch and atomic-write checks\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
