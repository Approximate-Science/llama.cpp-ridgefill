#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "RFLP requires IEEE float32");

inline void ridgefill_logprob_require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error("ridgefill-score: " + message); }
}

inline uint32_t ridgefill_logprob_u32(const uint8_t * bytes) {
    return uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 | uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
}

inline void ridgefill_logprob_put_u32(uint8_t * bytes, uint32_t value) {
    for (int i = 0; i < 4; ++i) { bytes[i] = uint8_t(value >> (8*i)); }
}

inline uint32_t ridgefill_logprob_read_u32(std::istream & file) {
    uint8_t bytes[4];
    file.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    ridgefill_logprob_require(bool(file), "--kl-against: truncated header");
    return ridgefill_logprob_u32(bytes);
}

inline void ridgefill_logprob_write_u32(std::ostream & file, uint32_t value) {
    uint8_t bytes[4];
    ridgefill_logprob_put_u32(bytes, value);
    file.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
}

struct ridgefill_logprob_reader {
    std::ifstream file;
    std::vector<uint8_t> bytes;

    ridgefill_logprob_reader(const std::string & path, uint32_t n_vocab, const std::vector<int32_t> & targets) :
            file(path, std::ios::binary | std::ios::ate) {
        ridgefill_logprob_require(bool(file), "--kl-against: cannot open " + path);
        const auto size = file.tellg();
        file.seekg(0);
        char magic[4];
        file.read(magic, sizeof(magic));
        ridgefill_logprob_require(bool(file) && std::memcmp(magic, "RFLP", 4) == 0, "--kl-against: invalid RFLP magic");
        ridgefill_logprob_require(ridgefill_logprob_read_u32(file) == 1, "--kl-against: unsupported version");
        ridgefill_logprob_require(ridgefill_logprob_read_u32(file) == 32, "--kl-against: dtype must be 32");
        ridgefill_logprob_require(ridgefill_logprob_read_u32(file) == n_vocab, "--kl-against: n_vocab mismatch");
        ridgefill_logprob_require(ridgefill_logprob_read_u32(file) == targets.size(), "--kl-against: n_scored mismatch");
        const uint64_t expected = 20 + uint64_t(targets.size())*(4 + uint64_t(n_vocab)*4);
        ridgefill_logprob_require(size >= 0 && uint64_t(size) == expected, "--kl-against: file size mismatch (truncated or trailing payload)");
        for (size_t i = 0; i < targets.size(); ++i) {
            ridgefill_logprob_require(ridgefill_logprob_read_u32(file) == uint32_t(targets[i]),
                    "--kl-against: target ids mismatch at scored position " + std::to_string(i));
        }
        ridgefill_logprob_require(n_vocab > 0 && n_vocab <= SIZE_MAX/4, "--kl-against: invalid n_vocab");
        bytes.resize(size_t(n_vocab)*4);
    }

    void read(std::vector<float> & row) {
        file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
        ridgefill_logprob_require(bool(file), "--kl-against: truncated payload");
        row.resize(bytes.size()/4);
        for (size_t i = 0; i < row.size(); ++i) {
            const uint32_t bits = ridgefill_logprob_u32(bytes.data() + 4*i);
            std::memcpy(&row[i], &bits, sizeof(bits));
            ridgefill_logprob_require(std::isfinite(row[i]) && row[i] <= 0, "--kl-against: invalid log-probability");
        }
    }
};

struct ridgefill_logprob_writer {
    std::string path, temporary;
    std::ofstream file;
    std::vector<uint8_t> bytes;
    size_t n_scored, n_written = 0;

    ridgefill_logprob_writer(const std::string & path, uint32_t n_vocab, const std::vector<int32_t> & targets) :
            path(path), temporary(path + ".tmp"), n_scored(targets.size()) {
        ridgefill_logprob_require(n_vocab > 0 && n_vocab <= SIZE_MAX/4 && !targets.empty() && targets.size() <= UINT32_MAX,
                "--save-logprobs: invalid dimensions");
        bytes.resize(size_t(n_vocab)*4);
        file.open(temporary, std::ios::binary | std::ios::trunc);
        ridgefill_logprob_require(bool(file), "--save-logprobs: cannot open " + temporary);
        file.write("RFLP", 4);
        ridgefill_logprob_write_u32(file, 1);
        ridgefill_logprob_write_u32(file, 32);
        ridgefill_logprob_write_u32(file, n_vocab);
        ridgefill_logprob_write_u32(file, uint32_t(targets.size()));
        for (int32_t target : targets) { ridgefill_logprob_write_u32(file, uint32_t(target)); }
    }

    ~ridgefill_logprob_writer() {
        if (!temporary.empty()) { file.close(); std::remove(temporary.c_str()); }
    }

    void write(const std::vector<float> & row) {
        ridgefill_logprob_require(row.size() == bytes.size()/4 && n_written < n_scored, "--save-logprobs: row count/size mismatch");
        for (size_t i = 0; i < row.size(); ++i) {
            ridgefill_logprob_require(std::isfinite(row[i]) && row[i] <= 0, "--save-logprobs: invalid log-probability");
            uint32_t bits;
            std::memcpy(&bits, &row[i], sizeof(bits));
            ridgefill_logprob_put_u32(bytes.data() + 4*i, bits);
        }
        file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        ridgefill_logprob_require(bool(file), "--save-logprobs: cannot write " + temporary);
        ++n_written;
    }

    void finish() {
        ridgefill_logprob_require(n_written == n_scored, "--save-logprobs: n_scored mismatch");
        file.close();
        ridgefill_logprob_require(bool(file), "--save-logprobs: cannot close " + temporary);
        ridgefill_logprob_require(std::rename(temporary.c_str(), path.c_str()) == 0, "--save-logprobs: cannot rename to " + path);
        temporary.clear();
    }
};

struct ridgefill_kl_statistics {
    std::vector<double> kl;
    std::vector<bool> top1;
    double dp_target_sum = 0;

    void add(const std::vector<float> & reference, const std::vector<float> & current, int32_t target) {
        ridgefill_logprob_require(!reference.empty() && reference.size() == current.size() && target >= 0 && size_t(target) < current.size(),
                "--kl-against: invalid comparison dimensions");
        double divergence = 0;
        size_t ref_top = 0, this_top = 0;
        for (size_t v = 0; v < reference.size(); ++v) {
            const double ref = reference[v], now = current[v];
            ridgefill_logprob_require(std::isfinite(ref) && std::isfinite(now) && ref <= 0 && now <= 0,
                    "--kl-against: invalid log-probability");
            divergence += std::exp(ref)*(ref - now);
            if (reference[v] > reference[ref_top]) { ref_top = v; }
            if (current[v] > current[this_top]) { this_top = v; }
        }
        kl.push_back(divergence);
        top1.push_back(ref_top == this_top);
        dp_target_sum += std::exp(double(current[target])) - std::exp(double(reference[target]));
    }

    double mean() const {
        double sum = 0;
        for (double value : kl) { sum += value; }
        return sum/kl.size();
    }

    double top1_agree() const {
        return double(std::count(top1.begin(), top1.end(), true))/top1.size();
    }

    double percentile(double fraction) const {
        auto sorted = kl;
        std::sort(sorted.begin(), sorted.end());
        const double position = fraction*(sorted.size() - 1);
        const size_t first = size_t(position), last = std::min(first + 1, sorted.size() - 1);
        return sorted[first] + (position - first)*(sorted[last] - sorted[first]);
    }
};
