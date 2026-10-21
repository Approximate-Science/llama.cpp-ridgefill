#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include "build-info.h"
#include "common.h"

#include "../src/llama-ext.h"
#include "../src/llama-model.h"
#include "json.h"

#include "fit.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
#include <sys/types.h>
#include <pwd.h>
#endif

#if defined(_AIX)
#include <sys/systemcfg.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

common_time_meas::common_time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

common_time_meas::~common_time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

//
// CPU utils
//

int32_t common_cpu_get_num_physical_cores() {
#if defined(_AIX)
    int32_t logical_cpus = _system_configuration.ncpus;
    int32_t smt_threads = _system_configuration.smt_threads;
    if (smt_threads > 0) {
        return static_cast<int32_t>(logical_cpus / smt_threads);
    }
    if (logical_cpus > 0) {
        return static_cast<int32_t>(logical_cpus);
    }
#elif defined(__linux__)
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    unsigned int n_threads_win = std::thread::hardware_concurrency();
    unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return default_threads;
        }
    }

    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
        return default_threads;
    }

    int32_t num_physical_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            num_physical_cores += info->Processor.GroupCount;
        }
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char*>(info) + info->Size);
    }

    return num_physical_cores > 0 ? num_physical_cores : default_threads;
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return !!(edx & (1u << 15));
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

static int cpu_count_math_cpus(int n_cpu) {
    int result = 0;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

#endif // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t common_cpu_get_num_math() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return common_cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#elif defined(__powerpc64__) || defined(__powerpc__)
    int32_t smt_factor = 1;
    int phy_cpus = common_cpu_get_num_physical_cores();
    int logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (phy_cpus > 0 && logical_cpus > phy_cpus) {
        smt_factor = logical_cpus / phy_cpus;
    }
    return phy_cpus * std::min(smt_factor, 2);
#endif
    return common_cpu_get_num_physical_cores();
}

// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    DWORD p = NORMAL_PRIORITY_CLASS;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = BELOW_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_NORMAL:   p = NORMAL_PRIORITY_CLASS;       break;
        case GGML_SCHED_PRIO_MEDIUM:   p = ABOVE_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_HIGH:     p = HIGH_PRIORITY_CLASS;         break;
        case GGML_SCHED_PRIO_REALTIME: p = REALTIME_PRIORITY_CLASS;     break;
    }

    if (!SetPriorityClass(GetCurrentProcess(), p)) {
        COM_WRN("failed to set process priority class %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#else // MacOS and POSIX
#include <sys/types.h>
#include <sys/resource.h>

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p =  5;  break;
        case GGML_SCHED_PRIO_NORMAL:   p =  0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = -5;  break;
        case GGML_SCHED_PRIO_HIGH:     p = -10; break;
        case GGML_SCHED_PRIO_REALTIME: p = -20; break;
    }

    if (setpriority(PRIO_PROCESS, 0, p) != 0) {
        COM_WRN("failed to set process priority %d : %s (%d)\n", prio, strerror(errno), errno);
        return false;
    }
    return true;
}

#endif

//
// CLI argument parsing
//


void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model) {
    int32_t n_set = 0;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model != nullptr) {
            cpuparams = *role_model;
        } else {
            cpuparams.n_threads = common_cpu_get_num_math();
        }
    }

    for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (cpuparams.cpumask[i]) {
            n_set++;
        }
    }

    if (n_set && n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        COM_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", n_set, cpuparams.n_threads);
    }
}

bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        COM_ERR("%s", "Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
        return false;
    }

    size_t start_i;
    size_t end_i;

    if (dash_loc == 0) {
        start_i = 0;
    } else {
        start_i = std::stoull(range.substr(0, dash_loc));
        if (start_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "Start index out of bounds!\n");
            return false;
        }
    }

    if (dash_loc == range.length() - 1) {
        end_i = GGML_MAX_N_THREADS - 1;
    } else {
        end_i = std::stoull(range.substr(dash_loc + 1));
        if (end_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "End index out of bounds!\n");
            return false;
        }
    }

    for (size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    size_t num_digits = mask.length() - start_i;
    num_digits = std::min<size_t>(num_digits, 128);

    size_t end_i = num_digits + start_i;

    for (size_t i = start_i, n = (num_digits*4 - 1); i < end_i; i++, n-=4) {
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            COM_ERR("Invalid hex character '%c' at position %d\n", c, int32_t(i));
            return false;
        }

        boolmask[  n  ] = boolmask[  n  ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

void common_init() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);

    llama_log_set(common_log_default_callback, NULL);
}

void common_params_print_info(const common_params & params, bool print_devices) {
#ifdef NDEBUG
    const char * build_type = "";
#else
    const char * build_type = " (debug)";
#endif
    COM_TRC("%s: build %d (%s) with %s for %s%s\n", __func__, llama_build_number(), llama_commit(), llama_compiler(), llama_build_target(), build_type);

    const int verbosity = common_log_get_verbosity_thold();
    COM_INF("%s: verbosity = %d (adjust with the `-lv N` CLI arg)\n", __func__, verbosity);

    // device enumeration creates a primary context on CUDA backends, skip it when the caller does not own any device
    if (print_devices && verbosity >= LOG_LEVEL_TRACE) {
        COM_TRC("%s", "device_info:\n");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            size_t free, total;
            ggml_backend_dev_memory(dev, &free, &total);
            COM_TRC("  - %-8s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
        }
    }
    COM_TRC("%s\n", common_params_get_system_info(params).c_str());
}

std::string common_params_get_system_info(const common_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.cpuparams.n_threads;
    if (params.cpuparams_batch.n_threads != -1) {
        os << " (n_threads_batch = " << params.cpuparams_batch.n_threads << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " | " << llama_print_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
#endif

    return os.str();
}

//
// String utils
//

std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string string_strip(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    return str.substr(start, end - start);
}

std::string string_lcs(std::string_view a, std::string_view b) {
    if (a.empty() || b.empty()) return {};

    std::vector<std::vector<size_t>> dp(a.size() + 1, std::vector<size_t>(b.size() + 1, 0));
    size_t best_len = 0;
    size_t best_end_a = 0;

    for (size_t i = 1; i <= a.size(); ++i) {
        for (size_t j = 1; j <= b.size(); ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if (dp[i][j] > best_len) {
                    best_len = dp[i][j];
                    best_end_a = i;
                }
            }
        }
    }
    return std::string(a.substr(best_end_a - best_len, best_len));
}

std::string string_get_sortable_timestamp() {
    using clock = std::chrono::system_clock;

    const clock::time_point current_time = clock::now();
    const time_t as_time_t = clock::to_time_t(current_time);
    char timestamp_no_ns[100];
    std::strftime(timestamp_no_ns, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&as_time_t));

    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        current_time.time_since_epoch() % 1000000000).count();
    char timestamp_ns[11];
    snprintf(timestamp_ns, 11, "%09" PRId64, ns);

    return std::string(timestamp_no_ns) + "." + std::string(timestamp_ns);
}

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string regex_escape(const std::string & s) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(s, special_chars, "\\$&");
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

std::string string_from(bool value) {
    return value ? "true" : "false";
}

std::string string_from(const std::vector<int> & values) {
    std::stringstream buf;

    buf << "[ ";
    bool first = true;
    for (auto e : values) {
        if (first) {
            first = false;
        } else {
            buf << ", ";
        }
        buf << std::to_string(e);
    }
    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, token);

        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }

    buf << " ]";

    return buf.str();
}

void string_process_escapes(std::string & input) {
    std::size_t input_len = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':  input[output_idx++] = '\n'; break;
                case 'r':  input[output_idx++] = '\r'; break;
                case 't':  input[output_idx++] = '\t'; break;
                case '\'': input[output_idx++] = '\''; break;
                case '\"': input[output_idx++] = '\"'; break;
                case '\\': input[output_idx++] = '\\'; break;
                case 'x':
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const char x[3] = { input[input_idx + 1], input[input_idx + 2], 0 };
                        char *err_p = nullptr;
                        const long val = std::strtol(x, &err_p, 16);
                        if (err_p == x + 2) {
                            input_idx += 2;
                            input[output_idx++] = char(val);
                            break;
                        }
                    }
                    // fall through
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr || sep - data >= 128) {
        COM_ERR("%s: malformed KV override '%s'\n", __func__, data);
        return false;
    }
    llama_model_kv_override kvo;
    std::strncpy(kvo.key, data, sep - data);
    kvo.key[sep - data] = 0;
    sep++;
    if (strncmp(sep, "int:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.val_i64 = std::atol(sep);
    } else if (strncmp(sep, "float:", 6) == 0) {
        sep += 6;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.val_f64 = std::atof(sep);
    } else if (strncmp(sep, "bool:", 5) == 0) {
        sep += 5;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (std::strcmp(sep, "true") == 0) {
            kvo.val_bool = true;
        } else if (std::strcmp(sep, "false") == 0) {
            kvo.val_bool = false;
        } else {
            COM_ERR("%s: invalid boolean value for KV override '%s'\n", __func__, data);
            return false;
        }
    } else if (strncmp(sep, "str:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (strlen(sep) > 127) {
            COM_ERR("%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
            return false;
        }
        strncpy(kvo.val_str, sep, 127);
        kvo.val_str[127] = '\0';
    } else {
        COM_ERR("%s: invalid type for KV override '%s'\n", __func__, data);
        return false;
    }
    overrides.emplace_back(std::move(kvo));
    return true;
}

static inline bool glob_class_match(const char c, const char * pattern, const char * class_end) {
    const char * class_start = pattern;
    bool negated = false;

    if (*class_start == '!') {
        negated = true;
        class_start++;
    }

    // If first character after negation is ']' or '-', treat it as literal
    if (*class_start == ']' || *class_start == '-') {
        if (class_start < class_end && *class_start == c) {
            return !negated;
        }
        class_start++;
    }

    bool matched = false;

    while (class_start < class_end) {
        if (class_start + 2 < class_end && class_start[1] == '-' && class_start[2] != ']') {
            char start_char = *class_start;
            char end_char = class_start[2];
            if (c >= start_char && c <= end_char) {
                matched = true;
                break;
            }
            class_start += 3;
        } else {
            if (*class_start == c) {
                matched = true;
                break;
            }
            class_start++;
        }
    }

    return negated ? !matched : matched;
}

// simple glob: * matches non-/ chars, ** matches anything including /, [] matches character class
static inline bool glob_match(const char * pattern, const char * str) {
    if (*pattern == '\0') {
        return *str == '\0';
    }
    if (pattern[0] == '*' && pattern[1] == '*') {
        const char * p = pattern + 2;
        if (glob_match(p, str)) return true;
        if (*str != '\0') return glob_match(pattern, str + 1);
        return false;
    }
    if (*pattern == '*') {
        const char * p = pattern + 1;
        for (; *str != '\0' && *str != '/'; str++) {
            if (glob_match(p, str)) return true;
        }
        return glob_match(p, str);
    }
    if (*pattern == '?' && *str != '\0' && *str != '/') {
        return glob_match(pattern + 1, str + 1);
    }
    if (*pattern == '[') {
        const char * class_end = pattern + 1;
        // If first character after '[' is ']' or '-', treat it as literal
        if (*class_end == ']' || *class_end == '-') {
            class_end++;
        }
        while (*class_end != '\0' && *class_end != ']') {
            class_end++;
        }
        if (*class_end == ']') {
            if (*str == '\0') return false;
            bool matched = glob_class_match(*str, pattern + 1, class_end);
            return matched && glob_match(class_end + 1, str + 1);
        } else {
            if (*str == '[') {
                return glob_match(pattern + 1, str + 1);
            }
            return false;
        }
    }
    if (*pattern == *str) {
        return glob_match(pattern + 1, str + 1);
    }
    return false;
}

bool glob_match(const std::string & pattern, const std::string & str) {
    return glob_match(pattern.c_str(), str.c_str());
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fs_validate_filename(const std::string & filename, bool allow_subdirs) {
    if (!filename.length()) {
        // Empty filename invalid
        return false;
    }
    if (filename.length() > 255) {
        // Limit at common largest possible filename on Linux filesystems
        // to avoid unnecessary further validation
        // (On systems with smaller limits it will be caught by the OS)
        return false;
    }

    size_t offset = 0;
    while (offset < filename.size()) {
        utf8_parse_result result = common_parse_utf8_codepoint(filename, offset);

        if (result.status != utf8_parse_result::SUCCESS) {
            return false;
        }
        uint32_t c = result.codepoint;

        if ((result.bytes_consumed == 2 && c < 0x80) ||
            (result.bytes_consumed == 3 && c < 0x800) ||
            (result.bytes_consumed == 4 && c < 0x10000)) {
            return false;
        }

        // Check for forbidden codepoints:
        // - Control characters
        // - Unicode equivalents of illegal characters
        // - UTF-16 surrogate pairs
        // - UTF-8 replacement character
        // - Byte order mark (BOM)
        // - Illegal characters: / \ : * ? " < > |
        if (c <= 0x1F // Control characters (C0)
            || c == 0x7F // Control characters (DEL)
            || (c >= 0x80 && c <= 0x9F) // Control characters (C1)
            || c == 0xFF0E // Fullwidth Full Stop (period equivalent)
            || c == 0x2215 // Division Slash (forward slash equivalent)
            || c == 0x2216 // Set Minus (backslash equivalent)
            || (c >= 0xD800 && c <= 0xDFFF) // UTF-16 surrogate pairs
            || c > 0x10FFFF // Max Unicode limit
            || c == 0xFFFD // Replacement Character (UTF-8)
            || c == 0xFEFF // Byte Order Mark (BOM)
            || c == ':' || c == '*' // Illegal characters
            || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            return false;
        }
        if (!allow_subdirs && (c == '/' || c == '\\')) {
            // Subdirectories not allowed, reject path separators
            return false;
        }
        offset += result.bytes_consumed;
    }

    // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
    // Unicode and other whitespace is not affected, only 0x20 space
    if (filename.front() == ' ' || filename.back() == ' ' || filename.back() == '.') {
        return false;
    }

    // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
    if (filename.find("..") != std::string::npos) {
        return false;
    }

    // Reject "."
    if (filename == ".") {
        return false;
    }

    return true;
}

#include <iostream>


#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string & str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}

std::string wstring_to_utf8(const std::wstring & str) {
    if (str.empty()) {
        return std::string();
    }

    int size = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0, NULL, NULL);

    if (size <= 0) {
        return std::string();
    }

    std::string utf8(size, 0);
    WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), &utf8[0], size, NULL, NULL);

    return utf8;
}
#endif

// returns the path as a UTF-8 string, preserving its separators
std::string fs_path_to_utf8(const std::filesystem::path & path) {
    const auto value = path.u8string();
    return std::string(value.begin(), value.end());
}

void fs_write_atomic(const std::filesystem::path & path, const std::string & data) {
    std::error_code ec;
    std::filesystem::path path_tmp = path;
    path_tmp += ".tmp";

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
    }

    std::ofstream file(path_tmp, std::ios::binary);
    file << data;
    file.close();

    if (!file.fail()) {
        std::filesystem::rename(path_tmp, path, ec);
    }

    if (file.fail() || ec) {
        std::filesystem::remove(path_tmp, ec);
        throw std::runtime_error("failed to write file: " + fs_path_to_utf8(path));
    }
}

bool fs_is_directory(const std::string & path) {
    std::filesystem::path dir(path);
    return std::filesystem::exists(dir) && std::filesystem::is_directory(dir);
}

std::string common_get_env(const std::string & name) {
    const char * value = std::getenv(name.c_str());
    return value == nullptr ? "" : value;
}

void common_set_env(const std::string & name, const std::string & value) {
#if defined(_WIN32)
    _putenv_s(name.c_str(), value.c_str());
#else
    if (value.empty()) {
        unsetenv(name.c_str());
    } else {
        setenv(name.c_str(), value.c_str(), 1);
    }
#endif
}

std::filesystem::path common_get_path_from_env(const std::string & name) {
#if defined(_WIN32)
    const std::wstring wname = utf8_to_wstring(name);
    const wchar_t * wvalue = _wgetenv(wname.c_str());
    return wvalue ? std::filesystem::path(wvalue) : std::filesystem::path();
#else
    const char * value = std::getenv(name.c_str());
    return value ? std::filesystem::path(value) : std::filesystem::path();
#endif
}

#if !defined(_WIN32)
static std::filesystem::path get_home_directory() {
    std::filesystem::path home = common_get_path_from_env("HOME");
    if (!home.empty()) {
        return home;
    }
    const struct passwd * pw = getpwuid(getuid());
    if (!pw || !pw->pw_dir || !*pw->pw_dir) {
        throw std::runtime_error("Failed to find $HOME directory");
    }
    return pw->pw_dir;
}
#endif

std::filesystem::path fs_get_cache_directory() {
    std::filesystem::path cache_directory = common_get_path_from_env("LLAMA_CACHE");
    if (!cache_directory.empty()) {
        return cache_directory;
    }
#if defined(_WIN32)
    cache_directory = common_get_path_from_env("LOCALAPPDATA");
    if (cache_directory.empty()) {
        throw std::runtime_error("Failed to find %LOCALAPPDATA% directory");
    }
#elif defined(__APPLE__)
    cache_directory = get_home_directory() / "Library/Caches";
#else
    cache_directory = common_get_path_from_env("XDG_CACHE_HOME");
    if (cache_directory.empty()) {
        cache_directory = get_home_directory() / ".cache";
    }
#endif
    return cache_directory / "llama.cpp";
}

std::filesystem::path fs_get_config_directory() {
    std::filesystem::path config_directory;
#if defined(_WIN32)
    config_directory = common_get_path_from_env("APPDATA");
    if (config_directory.empty()) {
        throw std::runtime_error("Failed to find %APPDATA% directory");
    }
#else
    config_directory = common_get_path_from_env("XDG_CONFIG_HOME");
    if (config_directory.empty()) {
        config_directory = get_home_directory() / ".config";
    }
#endif
    return config_directory / "llama.cpp";
}

std::filesystem::path fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    const std::filesystem::path cache_directory = fs_get_cache_directory();
    std::error_code ec;
    common_create_directories(cache_directory, ec);
    if (ec) {
        throw std::runtime_error("failed to create cache directory: " + fs_path_to_utf8(cache_directory));
    }
    return cache_directory / std::filesystem::u8path(filename);
}

//
// TTY utils
//

bool common_is_tty(FILE * file) {
#if defined(_WIN32)
    return _isatty(_fileno(file));
#else
    return isatty(fileno(file));
#endif
}

bool tty_can_use_colors() {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (const char * no_color = std::getenv("NO_COLOR")) {
        if (no_color[0] != '\0') {
            return false;
        }
    }

    // Check TERM environment variable
    if (const char * term = std::getenv("TERM")) {
        if (std::strcmp(term, "dumb") == 0) {
            return false;
        }
    }

    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    return common_is_tty(stdout) || common_is_tty(stderr);
}

bool tty_enable_ansi() {
#if defined(_WIN32)
    // a Windows console renders ANSI sequences only in virtual terminal mode, pipes and files take them as is
    for (DWORD id : { STD_OUTPUT_HANDLE, STD_ERROR_HANDLE }) {
        HANDLE h    = GetStdHandle(id);
        DWORD  mode = 0;
        if (GetConsoleMode(h, &mode) && !SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
            return false;
        }
    }
#endif
    return true;
}

//
// Model utils
//

// TODO: move to common/sampling
static void common_init_sampler_from_model(
    const llama_model * model,
    common_params_sampling & sparams) {

    const uint64_t config = sparams.user_sampling_config;

    auto get_int32 = [&](const char * key, int32_t & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[64] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            int32_t v = strtol(buf, &end, 10);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    auto get_float = [&](const char * key, float & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[128] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            float v = strtof(buf, &end);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    // Sampling sequence
    if (!(config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS)) {
        char buf[512] = {0};
        if (llama_model_meta_val_str(model, llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), buf, sizeof(buf)) > 0) {
            const std::vector<std::string> sampler_names = string_split<std::string>(std::string(buf), ';');
            if (!sampler_names.empty()) {
                sparams.samplers = common_sampler_types_from_names(sampler_names);
            }
        }
    }

    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K),           sparams.top_k,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P),           sparams.top_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P),           sparams.min_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY), sparams.xtc_probability, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD),   sparams.xtc_threshold,   common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TEMP),            sparams.temp,            common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N),  sparams.penalty_last_n,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT),  sparams.penalty_repeat,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT),        sparams.mirostat,        common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU),    sparams.mirostat_tau,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA),    sparams.mirostat_eta,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA);
}

struct common_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top

    common_threadpools threadpools;

    llama_model_ptr   model;

    // must outlive the context
    std::unique_ptr<llama_ridgefill, decltype(&llama_ridgefill_free)> ridgefill { nullptr, llama_ridgefill_free };

    llama_context_ptr context;

    std::vector<llama_adapter_lora_ptr> lora;

    std::vector<common_sampler_ptr> samplers;
    std::vector<llama_sampler_seq_config> samplers_seq_config;
};

static const std::map<common_decision_type, std::string> COMMON_DECISION_TYPE_NAMES = {
    { COMMON_DECISION_TYPE_OPENJEV,        "openjev"       },
    { COMMON_DECISION_TYPE_LEV,            "lev"           },
    { COMMON_DECISION_TYPE_KEV,            "kev"           },
    { COMMON_DECISION_TYPE_NIMBLE,         "nimble"        },
    { COMMON_DECISION_TYPE_LAYA,           "laya"          },
    { COMMON_DECISION_TYPE_CLEF,           "clef"          },
    { COMMON_DECISION_TYPE_PPLX_DECIDER,   "pplx-decider"  },
    { COMMON_DECISION_TYPE_VLLM_DECISION2, "vllm-decision2" },
    { COMMON_DECISION_TYPE_VLLM_D3,        "vllm-d3"        },
    { COMMON_DECISION_TYPE_LFM2_D1,        "lfm2-d1"       },
    { COMMON_DECISION_TYPE_LFM2_D1_OMNI,   "lfm2-d1-omni"  },
    { COMMON_DECISION_TYPE_GRANITE_SWITCH, "granite-switch"},
    { COMMON_DECISION_TYPE_H2O_LIGHTNING,  "h2o-lightning" },
    // for future models, ALWAYS add org name prefix to avoid name collisions, e.g. "orgname-modelname"
};

static common_decision_type common_decision_type_from_string(const std::string & str) {
    for (const auto & pair : COMMON_DECISION_TYPE_NAMES) {
        if (pair.second == str) {
            return pair.first;
        }
    }
    return COMMON_DECISION_TYPE_UNKNOWN;
}

common_decision_type common_get_decision_type(const struct llama_model * model) {
    char buf[64];
    if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) < 0) {
        return COMMON_DECISION_TYPE_NONE;
    }
    const std::string key = std::string(buf) + ".decision.type";
    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) < 0) {
        return COMMON_DECISION_TYPE_NONE;
    }
    return common_decision_type_from_string(buf);
}

common_gguf_info common_get_gguf_info(const std::string & fname) {
    common_gguf_info info;

    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(fname.c_str(), gguf_params));
    if (!gguf_ctx) {
        return info; // missing or unreadable file
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return info; // no architecture in the metadata
    }
    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch.empty()) {
        return info;
    }

    const int64_t type_id = gguf_find_key(gguf_ctx.get(), (arch + ".decision.type").c_str());
    if (type_id < 0) {
        info.decision_type = COMMON_DECISION_TYPE_NONE;
    } else if (gguf_get_kv_type(gguf_ctx.get(), type_id) == GGUF_TYPE_STRING) {
        info.decision_type = common_decision_type_from_string(gguf_get_val_str(gguf_ctx.get(), type_id));
    }

    // same key and type as the model loader
    const int64_t ctx_id = gguf_find_key(gguf_ctx.get(), (arch + ".context_length").c_str());
    if (ctx_id >= 0 && gguf_get_kv_type(gguf_ctx.get(), ctx_id) == GGUF_TYPE_UINT32) {
        info.n_ctx_train = gguf_get_val_u32(gguf_ctx.get(), ctx_id);
    }

    return info;
}

common_init_result::common_init_result(common_params & params, bool model_only) :
    pimpl(new impl{}) {
    if (!params.ridgefill_path.empty()) {
        if (params.split_mode == LLAMA_SPLIT_MODE_ROW) {
            COM_ERR("%s", "RidgeFill does not support --split-mode row; use --split-mode layer\n");
            return;
        }
        if (!params.lora_adapters.empty()) {
            COM_ERR("%s", "RidgeFill does not support LoRA (--lora)\n");
            return;
        }
        if (!params.control_vectors.empty()) {
            COM_ERR("%s", "RidgeFill does not support control vectors (--control-vector)\n");
            return;
        }
        if (params.embedding) {
            COM_ERR("%s", "RidgeFill does not support embeddings (--embeddings)\n");
            return;
        }
        if (params.n_cache_reuse != 0) {
            COM_ERR("%s", "RidgeFill does not support --cache-reuse; set --cache-reuse 0\n");
            return;
        }
    }
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (params.fit_params) {
        COM_TRC("%s", "fitting params to device memory ...\n");
        COM_TRC("%s", "(for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)\n");

        // the draft context is created from the same base params and follows the main context, fit both together
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp  = std::find(params.speculative.types.begin(), params.speculative.types.end(),
            COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

        common_params params_dft = common_base_params_to_speculative(params);

        auto mparams_dft = common_model_params_to_llama(params_dft);
        auto cparams_dft = common_context_params_to_llama(params_dft);
        if (spec_mtp) {
            cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        }
        cparams_dft.n_rs_seq = 0;

        const common_fit_extra_model extra = {
            /*.path_model   =*/ params_dft.model.path.c_str(),
            /*.mparams      =*/ &mparams_dft,
            /*.cparams      =*/ &cparams_dft,
            /*.shares_model =*/ !has_draft, // an MTP context runs on the weights of the main model
        };

        common_fit_params(params.model.path.c_str(), &mparams, &cparams,
            params.tensor_split,
            params.tensor_buft_overrides.data(),
            params.fit_params_target.data(),
            params.fit_params_min_ctx,
            has_draft || spec_mtp ? &extra : nullptr,
            params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
    }

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == NULL) {
        return;
    }

    pimpl->model.reset(model);

    if (model_only) {
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // these decision models return a score for each token via the embeddings output
    // TODO: maybe improve this in the future
    const auto decision_type = common_get_decision_type(model);
    if (decision_type == COMMON_DECISION_TYPE_LAYA || decision_type == COMMON_DECISION_TYPE_KEV || decision_type == COMMON_DECISION_TYPE_CLEF ||
        decision_type == COMMON_DECISION_TYPE_LFM2_D1_OMNI || decision_type == COMMON_DECISION_TYPE_VLLM_DECISION2) {
        params.embedding    = true;
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;

        cparams.embeddings            = true;
        cparams.pooling_type          = LLAMA_POOLING_TYPE_NONE;
        cparams.n_outputs_max         = cparams.n_batch;
        cparams.n_outputs_max_per_seq = 1;

        LOG_INF("%s", "decision model reads the embeddings output, enabling embedding mode\n");
    }

    // embeddings need the whole batch in one ubatch, so n_batch must not be larger than n_ubatch
    // (server.cpp does this check for --embedding, but before the model is loaded)
    if (cparams.embeddings && cparams.n_batch > cparams.n_ubatch) {
        LOG_WRN("embeddings enabled: setting n_batch = n_ubatch = %u\n", cparams.n_ubatch);
        cparams.n_batch = cparams.n_ubatch;
        params.n_batch  = params.n_ubatch;
    }

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_adapter_lora_ptr lora;
        lora.reset(llama_adapter_lora_init(model, la.path.c_str()));
        if (lora == nullptr) {
            COM_ERR("failed to load lora adapter '%s'\n", la.path.c_str());
            return;
        }

        char buf[1024];
        la.ptr = lora.get();
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.task_name", buf, sizeof(buf));
        la.task_name = buf;
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.prompt_prefix", buf, sizeof(buf));
        la.prompt_prefix = buf;
        pimpl->lora.emplace_back(std::move(lora)); // copy to list of loaded adapters
    }

    // updates params.sampling
    // TODO: fix naming
    common_init_sampler_from_model(model, params.sampling);

    if (params.sampling.ignore_eos && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        COM_WRN("%s", "vocab does not have an EOS token, ignoring --ignore-eos\n");
        params.sampling.ignore_eos = false;
    }

    // initialize once
    for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            COM_TRC("added %s logit bias = %f\n", common_token_to_piece(vocab, i).c_str(), -INFINITY);
            params.sampling.logit_bias_eog.push_back({i, -INFINITY});
        }
    }

    if (params.sampling.ignore_eos) {
        // add EOG biases to the active set of logit biases
        params.sampling.logit_bias.insert(
                params.sampling.logit_bias.end(),
                params.sampling.logit_bias_eog.begin(), params.sampling.logit_bias_eog.end());
    }

    // init the backend samplers as part of the context creation
    pimpl->samplers.resize(cparams.n_seq_max);
    pimpl->samplers_seq_config.resize(cparams.n_seq_max);

    for (int i = 0; i < (int) cparams.n_seq_max; ++i) {
        pimpl->samplers[i].reset(common_sampler_init(model, params.sampling));
        pimpl->samplers_seq_config[i] = { i, common_sampler_get(pimpl->samplers[i].get()) };
    }

    if (params.sampling.backend_sampling) {
        cparams.samplers   = pimpl->samplers_seq_config.data();
        cparams.n_samplers = pimpl->samplers_seq_config.size();
    }

    llama_context * lctx = llama_init_from_model(model, cparams);
    if (lctx == NULL) {
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return;
    }

    pimpl->context.reset(lctx);

    if (!params.ridgefill_path.empty()) {
        if (cparams.embeddings) {
            COM_ERR("%s", "RidgeFill does not support embeddings enabled by the model\n");
            pimpl->context.reset();
            return;
        }

        pimpl->ridgefill.reset(llama_ridgefill_init(model, params.ridgefill_path.c_str()));
        if (!pimpl->ridgefill) {
            COM_ERR("failed to load RidgeFill projector '%s'\n", params.ridgefill_path.c_str());
            pimpl->context.reset();
            return;
        }

        try {
            common_ridgefill_resolve(params, pimpl->ridgefill.get());
        } catch (const std::exception & error) {
            COM_ERR("RidgeFill: %s\n", error.what());
            pimpl->context.reset();
            return;
        }
        llama_set_ridgefill(lctx, pimpl->ridgefill.get());
        if (params.ridgefill_mode != "off" && llama_ridgefill_set_alpha(lctx, params.ridgefill_alpha) < 0) {
            COM_ERR("%s", "RidgeFill: --ridgefill-alpha was refused\n");
            pimpl->context.reset();
            return;
        }
        COM_INF("RidgeFill: %s, mode %s, split %d, exact tail %d, min prompt %d\n",
                llama_ridgefill_desc(pimpl->ridgefill.get()), params.ridgefill_mode.c_str(),
                llama_ridgefill_split(pimpl->ridgefill.get()), params.ridgefill_tail, params.ridgefill_min_prompt);
    }

    set_process_priority(params.cpuparams.priority);

    pimpl->threadpools.init(lctx, params);
}

llama_model * common_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_init_result::context() {
    return pimpl->context.get();
}

llama_ridgefill * common_init_result::ridgefill() {
    return pimpl->ridgefill.get();
}

common_sampler * common_init_result::sampler(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= (int) pimpl->samplers.size()) {
        return nullptr;
    }
    return pimpl->samplers[seq_id].get();
}

void common_init_result::reset_samplers() {
    for (int i = 0; i < (int) pimpl->samplers.size(); ++i) {
        llama_sampler_reset(common_sampler_get(pimpl->samplers[i].get()));
    }
}

std::vector<llama_adapter_lora_ptr> & common_init_result::lora() {
    return pimpl->lora;
}

// only for warmup and probe decodes, fill zeros as dummy input
static void common_batch_set_zero_state(common_batch & batch, const llama_model * model, std::vector<float> & zeros) {
    zeros.assign(llama_model_n_embd_out(model), 0.0f);
    for (int32_t i = 0; i < batch.size(); ++i) {
        batch.set_embd_state(i, { zeros.data(), 1, zeros.size() });
    }
}

common_init_result_ptr common_init_from_params(common_params & params, bool model_only) {
    common_init_result_ptr res(new common_init_result(params, model_only));

    llama_model * model = res->model();
    if (model == NULL) {
        COM_ERR("failed to load model '%s'\n", params.model.path.c_str());
        return res;
    }

    if (model_only) {
        return res;
    }

    llama_context * lctx = res->context();
    if (lctx == NULL) {
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return res;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (params.ctx_shift && !llama_memory_can_shift(llama_get_memory(lctx))) {
        COM_WRN("%s", "KV cache shifting is not supported for this context, disabling KV cache shifting\n");
        params.ctx_shift = false;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_model_n_layer(model);

        const auto cvec = common_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            return res;
        }

        int err = llama_set_adapter_cvec(
                lctx,
                cvec.data.data(),
                cvec.data.size(),
                cvec.n_embd,
                params.control_vector_layer_start,
                params.control_vector_layer_end);
        if (err) {
            return res;
        }
    }

    if (llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_RANK) {
        bool ok = true;

        if (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL) {
            COM_WRN("%s", "vocab does not have a  BOS token, reranking will not work\n");
            ok = false;
        }

        bool has_eos = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
        bool has_sep = llama_vocab_sep(vocab) != LLAMA_TOKEN_NULL;
        bool has_rerank_prompt = llama_model_chat_template(model, "rerank") != NULL;

        if (!has_eos && !has_sep && !has_rerank_prompt) {
            COM_WRN("%s", "vocab does not have an EOS token, SEP token, or rerank prompt. Reranking will not work\n");
            ok = false;
        } else if (!has_eos) {
            COM_WRN("%s", "vocab does not have an EOS token, using SEP token as fallback\n");
        }

        if (!ok) {
            return res;
        }
    }

    if (!params.lora_init_without_apply) {
        common_set_adapter_lora(lctx, params.lora_adapters);
    }

    if (params.warmup) {
        COM_TRC("%s", "warming up the model with an empty run - please wait ... (--no-warmup to disable)\n");

        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);

        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            common_batch batch = common_batch_get_one(lctx, tmp);
            llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get());
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            tmp.resize(std::min(tmp.size(), (size_t) params.n_batch));
            common_batch batch = common_batch_get_one(lctx, tmp);
            std::vector<float> zeros;
            common_batch_set_zero_state(batch, model, zeros);
            llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        }
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);

        // reset samplers to reset RNG state after warmup to the seeded state
        res->reset_samplers();
    }

    return res;
}

common_init_result::~common_init_result() = default;

std::string common_get_model_endpoint() {
    std::string endpoint = common_get_env("MODEL_ENDPOINT");
    if (endpoint.empty()) {
        // the HF_ENDPOINT variable is respected for backward compatibility
        endpoint = common_get_env("HF_ENDPOINT");
    }
    if (endpoint.empty()) {
        return "https://huggingface.co/";
    }
    if (endpoint.back() != '/') {
        endpoint += '/';
    }
    return endpoint;
}

char * common_get_model_or_exit(int argc, char * argv[]) {
    if (argc > 1) {
        return argv[1];
    }

    char * path = getenv("LLAMACPP_TEST_MODELFILE");
    if (!path || strlen(path) == 0) {
        fprintf(stderr, "\033[33mWARNING: No model file provided. Skipping this test. Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> to silence this warning and run this test.\n\033[0m");
        exit(EXIT_SUCCESS);
    }

    return path;
}

common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx) {
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    if (llama_n_rs_seq(ctx) > 0) {
        COM_TRC("%s", "the context supports bounded partial sequence removal\n");
        return COMMON_CONTEXT_SEQ_RM_TYPE_RS;
    }

    common_context_seq_rm_type res = COMMON_CONTEXT_SEQ_RM_TYPE_PART;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret;
    {
        common_batch batch = common_batch_get_one(ctx, tmp);
        std::vector<float> zeros;
        common_batch_set_zero_state(batch, llama_get_model(ctx), zeros);
        ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    }
    if (ret != 0) {
        COM_ERR("llama_process() failed: %d\n", ret);
        res = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        COM_TRC("%s", "the context does not support partial sequence removal\n");
        res = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx);

    return res;
}

static void common_context_seq_rm(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    if (!llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        GGML_ABORT("%s", string_format("failed to remove sequence %d with p0=%d, p1=%d\n", seq_id, p0, p1).c_str());
    }
}

static void common_context_seq_cp(llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
}

static void common_context_seq_add(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_add(mem, seq_id, p0, p1, delta);
}

void common_memory::init(llama_context * ctx_tgt, llama_context * ctx_dft) {
    this->ctx_tgt = ctx_tgt;
    this->ctx_dft = ctx_dft;
}

void common_memory::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    common_context_seq_rm(ctx_tgt, seq_id, p0, p1);
    if (ctx_dft) {
        common_context_seq_rm(ctx_dft, seq_id, p0, p1);
    }
}

void common_memory::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) const {
    common_context_seq_cp(ctx_tgt, seq_id_src, seq_id_dst, p0, p1);
    if (ctx_dft) {
        common_context_seq_cp(ctx_dft, seq_id_src, seq_id_dst, p0, p1);
    }
}

void common_memory::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) const {
    common_context_seq_add(ctx_tgt, seq_id, p0, p1, delta);
    if (ctx_dft) {
        common_context_seq_add(ctx_dft, seq_id, p0, p1, delta);
    }
}

int32_t common_ridgefill_p_end(const common_params & params, int32_t n_prompt, int32_t n_tail, const std::string & mode) {
    if (n_tail <= 0) {
        n_tail = params.ridgefill_tail;
    }

    if (params.ridgefill_path.empty() || (mode.empty() ? params.ridgefill_mode : mode) == "off" || n_prompt < params.ridgefill_min_prompt) {
        return 0;
    }
    if (n_tail <= 0) { throw std::invalid_argument("RidgeFill: projector tail_default has not been resolved"); }
    return std::max(0, n_prompt - n_tail);
}

bool common_ridgefill_mode_valid(const std::string & mode) {
    return mode == "off" || mode == "speed" || mode == "quality" || mode == "quality+qattn";
}

void common_ridgefill_set_option(common_params & params, const std::string & option, const std::string & value) {
    auto number = [&]() {
        size_t end = 0;
        float result;
        try {
            result = std::stof(value, &end);
        } catch (const std::exception &) {
            throw std::invalid_argument(option + " must be finite and in [0, 1]");
        }
        if (end != value.size() || !std::isfinite(result) || result < 0 || result > 1) {
            throw std::invalid_argument(option + " must be finite and in [0, 1]");
        }
        return result;
    };
    if (option == "--ridgefill-mode") {
        if (!common_ridgefill_mode_valid(value)) {
            throw std::invalid_argument("--ridgefill-mode must be off, speed, quality or quality+qattn");
        }
        params.ridgefill_mode = value;
    } else if (option == "--ridgefill-qattn-window-mode" || option == "--qattn-window-mode") {
        if (value != "last" && value != "turn") { throw std::invalid_argument(option + " must be last or turn"); }
        params.ridgefill_qattn_window_mode = value;
    } else if (option == "--ridgefill-header-rows") {
        size_t end = 0;
        int result;
        try { result = std::stoi(value, &end); }
        catch (const std::exception &) { throw std::invalid_argument(option + " must be a nonnegative integer"); }
        if (end != value.size() || result < 0) { throw std::invalid_argument(option + " must be a nonnegative integer"); }
        params.ridgefill_header_rows = result;
    } else if (option == "--ridgefill-s1-factor") {
        size_t end = 0;
        double factor;
        try { factor = std::stod(value, &end); }
        catch (const std::exception &) { throw std::invalid_argument(option + " must be finite and nonnegative"); }
        if (end != value.size() || !std::isfinite(factor) || factor < 0) {
            throw std::invalid_argument(option + " must be finite and nonnegative");
        }
        params.ridgefill_s1_factor = factor;
    } else if (option == "--ridgefill-s1-window" || option == "--ridgefill-vtail-rows") {
        size_t end = 0;
        int count;
        const int minimum = option == "--ridgefill-s1-window" ? 0 : 1;
        try { count = std::stoi(value, &end); }
        catch (const std::exception &) { throw std::invalid_argument(option + " has an invalid row count"); }
        if (end != value.size() || count < minimum) { throw std::invalid_argument(option + " has an invalid row count"); }
        if (option == "--ridgefill-s1-window") { params.ridgefill_s1_window = count; }
        else { params.ridgefill_vtail_rows = count; }
    } else if (option == "--ridgefill-alpha") {
        params.ridgefill_alpha = number();
    } else if (option == "--ridgefill-share") {
        params.ridgefill_share = number();
    } else if (option == "--ridgefill-qattn-share") {
        params.ridgefill_qattn_share = number();
    } else if (option == "--ridgefill-digits" || option == "--ridgefill-qattn-blind" || option == "--ridgefill-special-exact" || option == "--ridgefill-vtail") {
        if (value != "on" && value != "off") { throw std::invalid_argument(option + " must be on or off"); }
        if (option == "--ridgefill-digits") {
            params.ridgefill_digits = value == "on";
        } else if (option == "--ridgefill-vtail") {
            params.ridgefill_vtail = value == "on";
        } else if (option == "--ridgefill-special-exact") {
            params.ridgefill_special_exact = value == "on";
        } else {
            params.ridgefill_qattn_blind = value == "on";
        }
    } else if (option == "--ridgefill-qattn-window" || option == "--ridgefill-tail" || option == "--ridgefill-qattn-turn-cap" || option == "--qattn-turn-cap") {
        size_t end = 0;
        int result;
        try {
            result = std::stoi(value, &end);
        } catch (const std::exception &) {
            throw std::invalid_argument(option + " must be a positive integer");
        }
        if (end != value.size() || result < 1) { throw std::invalid_argument(option + " must be a positive integer"); }
        if (option == "--ridgefill-tail") {
            params.ridgefill_tail = result;
        } else if (option == "--ridgefill-qattn-turn-cap" || option == "--qattn-turn-cap") {
            params.ridgefill_qattn_turn_cap = result;
        } else {
            params.ridgefill_qattn_window = result;
        }
    } else {
        throw std::invalid_argument("unknown RidgeFill option " + option);
    }
}

static void common_ridgefill_check_quality_alpha(const common_params & params, const std::string & mode) {
    if ((mode == "quality" || mode == "quality+qattn") && params.ridgefill_alpha != 0) {
        throw std::invalid_argument("RidgeFill: mode " + mode + " with nonzero --ridgefill-alpha requires the rho-scaled correction; use --ridgefill-alpha 0");
    }
}

void common_ridgefill_validate_options(const common_params & params, const std::string & mode) {
    const auto & selected_mode = mode.empty() ? params.ridgefill_mode : mode;
    if (!common_ridgefill_mode_valid(selected_mode)) { throw std::invalid_argument("RidgeFill: invalid mode " + selected_mode); }
    if (params.ridgefill_header_rows < 0) { throw std::invalid_argument("RidgeFill: --ridgefill-header-rows must be nonnegative"); }
    if (params.ridgefill_qattn_window_mode != "last" && params.ridgefill_qattn_window_mode != "turn") {
        throw std::invalid_argument("RidgeFill: --ridgefill-qattn-window-mode must be last or turn");
    }
    if (selected_mode == "speed" && (params.ridgefill_special_exact || params.ridgefill_header_rows > 0)) {
        throw std::invalid_argument("RidgeFill: --ridgefill-special-exact and --ridgefill-header-rows require quality or quality+qattn, not mode speed");
    }
    if (params.ridgefill_qattn_turn_cap <= 0) { throw std::invalid_argument("RidgeFill: --qattn-turn-cap must be positive"); }
    if (selected_mode != "off" && params.ridgefill_qattn_window_mode == "turn") {
        if (selected_mode != "quality+qattn") { throw std::invalid_argument("RidgeFill: --qattn-window-mode turn requires quality+qattn"); }
        if (params.ridgefill_qattn_blind) { throw std::invalid_argument("RidgeFill: --qattn-window-mode turn does not support --ridgefill-qattn-blind"); }
    }
    if (params.ridgefill_s1_window < 0) { throw std::invalid_argument("RidgeFill: --ridgefill-s1-window must be nonnegative"); }
    if (!std::isfinite(params.ridgefill_s1_factor) || params.ridgefill_s1_factor < 0) {
        throw std::invalid_argument("RidgeFill: --ridgefill-s1-factor must be finite and nonnegative");
    }
    if (params.ridgefill_vtail_rows < 1) { throw std::invalid_argument("RidgeFill: --ridgefill-vtail-rows must be positive"); }
    if (selected_mode != "off" && (params.ridgefill_s1_window > 0 || params.ridgefill_vtail)) {
        if (selected_mode != "quality+qattn") {
            throw std::invalid_argument("RidgeFill: S1/vtail requires the quality+qattn two-sweep flow");
        }
        if (params.speculative.need_n_rs_seq() > 0) {
            throw std::invalid_argument("RidgeFill: S1/vtail does not support recurrent rollback snapshots (n_rs_seq > 0)");
        }
        if (params.attention_type == LLAMA_ATTENTION_TYPE_NON_CAUSAL) {
            throw std::invalid_argument("RidgeFill: S1/vtail does not support non-causal contexts");
        }
        if (params.ridgefill_s1_window > 0 && params.ridgefill_header_rows > 0) {
            throw std::invalid_argument("RidgeFill: S1 base selection does not support --ridgefill-header-rows");
        }
        if (params.ridgefill_s1_window > 0 && params.ridgefill_qattn_share == 0) {
            throw std::invalid_argument("RidgeFill: S1 base selection requires positive --ridgefill-qattn-share");
        }
        if (params.ridgefill_s1_window > 0 && params.ridgefill_qattn_blind) {
            throw std::invalid_argument("RidgeFill: S1 does not support --ridgefill-qattn-blind (window overlaps the prefix)");
        }
    }
    if (!params.ridgefill_path.empty()) {
        if (params.split_mode == LLAMA_SPLIT_MODE_ROW) { throw std::invalid_argument("RidgeFill does not support --split-mode row; use --split-mode layer"); }
        if (!params.lora_adapters.empty()) { throw std::invalid_argument("RidgeFill does not support LoRA (--lora)"); }
        if (!params.control_vectors.empty()) { throw std::invalid_argument("RidgeFill does not support control vectors (--control-vector)"); }
        if (params.embedding) { throw std::invalid_argument("RidgeFill does not support embeddings (--embeddings)"); }
        if (params.n_cache_reuse != 0) { throw std::invalid_argument("RidgeFill does not support --cache-reuse; use --cache-reuse 0"); }
    }
}

void common_ridgefill_require_api_v2(const common_params & params, const std::string & mode) {
    const auto & selected_mode = mode.empty() ? params.ridgefill_mode : mode;
    if (selected_mode == "off" || (params.ridgefill_s1_window == 0 && !params.ridgefill_vtail)) { return; }
#if defined(LLAMA_RIDGEFILL_API_VERSION) && LLAMA_RIDGEFILL_API_VERSION >= 2
    if (llama_ridgefill_api_version() >= 2) { return; }
#endif
    throw std::invalid_argument("RidgeFill: S1/vtail requires C API v2; unavailable in this build");
}

void common_ridgefill_validate_sweep(const common_params & params, int32_t n_prompt, int32_t n_prefix,
        int32_t selector_end, int32_t n_ubatch) {
    common_ridgefill_validate_options(params);
    if (params.ridgefill_mode == "off" || (params.ridgefill_s1_window == 0 && !params.ridgefill_vtail)) { return; }
    if (n_prefix <= 0 || n_prefix >= n_prompt) { throw std::invalid_argument("RidgeFill: S1/vtail requires a nonempty filled prefix and exact tail"); }
    if (params.ridgefill_s1_window > 0 && (selector_end > n_prompt || selector_end <= n_prefix ||
            params.ridgefill_s1_window > selector_end - n_prefix)) {
        throw std::invalid_argument("RidgeFill: S1 window must be nonempty, within LOWER rows and not overlap the prefix");
    }
    if (params.ridgefill_vtail && std::min(n_ubatch, params.n_batch) < std::min(params.ridgefill_vtail_rows, n_prompt - n_prefix)) {
        throw std::invalid_argument("RidgeFill: vtail first tail ubatch must hold min(--ridgefill-vtail-rows, exact tail) rows; increase --ubatch-size and --batch-size");
    }
}

void common_ridgefill_resolve(common_params & params, const llama_ridgefill * rf) {
    if (!common_ridgefill_mode_valid(params.ridgefill_mode)) {
        throw std::invalid_argument("invalid --ridgefill-mode " + params.ridgefill_mode);
    }
    if (params.ridgefill_tail == 0) { params.ridgefill_tail = llama_ridgefill_tail_default(rf); }
    if (params.ridgefill_alpha < 0) { params.ridgefill_alpha = llama_ridgefill_alpha_default(rf); }
    if (params.ridgefill_share < 0) { params.ridgefill_share = llama_ridgefill_share_default(rf); }
    if (params.ridgefill_digits < 0) { params.ridgefill_digits = llama_ridgefill_digits_default(rf); }
    if (params.ridgefill_qattn_share < 0) { params.ridgefill_qattn_share = llama_ridgefill_qattn_default(rf); }
    if (params.ridgefill_tail < llama_ridgefill_tail_min(rf)) {
        throw std::invalid_argument("--ridgefill-tail is below this projector's tail_min");
    }
    common_ridgefill_check_quality_alpha(params, params.ridgefill_mode);
    common_ridgefill_validate_options(params);
    common_ridgefill_require_api_v2(params);
}

std::vector<llama_pos> common_ridgefill_rows(const common_params & params, const llama_ridgefill * rf,
        const llama_token * tokens, int32_t n_prefix, const std::string & mode) {
    const auto & selected_mode = mode.empty() ? params.ridgefill_mode : mode;
    common_ridgefill_validate_options(params, selected_mode);
    if (selected_mode == "off" || selected_mode == "speed" || n_prefix == 0) { return {}; }
    common_ridgefill_check_quality_alpha(params, selected_mode);
    if (!rf || !tokens || n_prefix < 0) { throw std::invalid_argument("RidgeFill: class/digit selection requires a projector, tokens and nonnegative prefix length"); }
    if (!(params.ridgefill_share >= 0 && params.ridgefill_share <= 1)) {
        throw std::invalid_argument("RidgeFill: --ridgefill-share must be resolved and in [0, 1]");
    }
    if (params.ridgefill_digits != 0 && params.ridgefill_digits != 1) {
        throw std::invalid_argument("RidgeFill: --ridgefill-digits must be resolved to on or off");
    }
    std::vector<llama_pos> rows(n_prefix);
    auto selection = llama_ridgefill_select_default_params(rf);
    selection.share = params.ridgefill_share;
    selection.digits = params.ridgefill_digits != 0;
    selection.special_exact = params.ridgefill_special_exact;
    selection.header_rows = std::min(params.ridgefill_header_rows, n_prefix);
    const int32_t count = llama_ridgefill_select_rows_ext(rf, tokens, n_prefix, &selection, rows.data());
    if (count < 0 || count > n_prefix) { throw std::runtime_error("RidgeFill: llama_ridgefill_select_rows_ext refused mode " + selected_mode); }
    rows.resize(count);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] < 0 || rows[i] >= n_prefix || (i > 0 && rows[i] <= rows[i - 1])) {
            throw std::runtime_error("RidgeFill: llama_ridgefill_select_rows_ext returned invalid prefix positions");
        }
    }
    return rows;
}

common_json common_ridgefill_report::to_json() const {
    const int32_t filled = n_prefix - n_exact_prefix;
    common_json result {
        {"n_prompt", n_prompt}, {"n_prefix", n_prefix}, {"n_filled", filled},
        {"n_exact_prefix", n_exact_prefix}, {"n_cached_exact_prefix", n_cached_exact_prefix}, {"n_exact_tail", n_prompt - n_prefix},
        {"n_exact_prefix_by_source", {{"class", exact_by_source[0]}, {"digit", exact_by_source[1]},
            {"special", exact_by_source[2]}, {"header", exact_by_source[3]}, {"qattn", exact_by_source[4]}}},
        {"tokens_skipped", n_prompt > 0 ? double(filled) / n_prompt : 0.0},
        {"macs_exact", macs_exact}, {"macs_ridgefill", macs_ridgefill},
        {"mac_share_skipped", macs_exact > 0 ? 1.0 - macs_ridgefill / macs_exact : 0.0},
        {"scope", "prompt coverage including cached rows; analytic cold prefill"},
        {"mac_model", "kva/compute.py: efficient bypass, actual exact-row positions, actual query window, one lm_head"},
    };
    if (s1_begin >= 0) {
        result["n_exact_prefix_by_source"]["s1"] = exact_by_source[5];
        result["s1"] = common_json {{"window", std::vector<int32_t>{s1_begin, s1_end}}, {"macs_fill", macs_s1_fill},
            {"macs_probe", macs_s1_probe}, {"probe_kv_pairs", s1_probe_kv_pairs}, {"fill_recomputed", true}};
    }
    if (vtail_rows > 0) { result["vtail"] = common_json {{"rows", vtail_rows}, {"macs_measure", macs_vtail}}; }
    if (s1_begin >= 0 || vtail_rows > 0) {
        result["mac_model"] = "kva cold prefill + S1 dense probe pairs and recomputed prefix projector/KV + vtail projector and two V projections";
    }
    return result;
}

common_ridgefill_report common_ridgefill_get_report(const llama_model * model, const llama_ridgefill * rf,
        const common_params & params, const llama_token * tokens, int32_t n_prompt, int32_t n_prefix,
        const std::vector<llama_pos> & selected_rows, const std::string & mode, int32_t window_begin, int32_t window_end,
        int32_t filled_begin, int32_t filled_end, const common_ridgefill_sweep_report * sweep) {
    auto require = [](bool ok, const char * message) {
        if (!ok) { throw std::runtime_error(std::string("RidgeFill reporting: ") + message); }
    };
    require(model && n_prompt >= 0 && n_prefix >= 0 && n_prefix <= n_prompt, "invalid prompt bounds");
    require(common_ridgefill_mode_valid(mode), "invalid mode");
    common_ridgefill_validate_options(params, mode);
    require(mode != "off" || n_prefix == 0, "off mode cannot fill prefix rows");
    require(selected_rows.size() <= size_t(n_prefix), "too many exact prefix rows");
    for (size_t i = 0; i < selected_rows.size(); ++i) {
        require(selected_rows[i] >= 0 && selected_rows[i] < n_prefix && (i == 0 || selected_rows[i] > selected_rows[i - 1]),
                "invalid exact prefix rows");
    }
    if (filled_end < 0) { filled_end = n_prefix; }
    require(filled_begin >= 0 && filled_begin <= filled_end && filled_end <= n_prefix, "invalid cached fill bounds");
    std::vector<llama_pos> rows = selected_rows;
    if (filled_begin > 0 || filled_end < n_prefix) {
        rows.clear();
        size_t next = 0;
        for (int32_t i = 0; i < n_prefix; ++i) {
            const bool selected = next < selected_rows.size() && selected_rows[next] == i;
            if (i < filled_begin || i >= filled_end || selected) { rows.push_back(i); }
            if (selected) { ++next; }
        }
    }
    common_ridgefill_report report;
    report.enabled = true;
    report.n_prompt = n_prompt;
    report.n_prefix = n_prefix;
    report.n_exact_prefix = rows.size();
    const bool s1 = mode != "off" && params.ridgefill_s1_window > 0;
    const bool vtail = mode != "off" && params.ridgefill_vtail;
    if (s1 || vtail) {
        require(rf && tokens && model->arch == LLM_ARCH_QWEN35, "S1/vtail requires qwen35 and a projector");
        require(sweep != nullptr, "S1/vtail needs completed sweep metadata");
        require(n_prefix > 0 && n_prefix < n_prompt, "S1/vtail needs a nonempty prefix and exact tail");
        if (s1) {
            require(sweep->s1_begin >= n_prefix && sweep->s1_end <= n_prompt &&
                    int64_t(sweep->s1_end) - sweep->s1_begin == params.ridgefill_s1_window, "invalid completed S1 window");
            report.s1_begin = sweep->s1_begin;
            report.s1_end = sweep->s1_end;
            const uint64_t width = params.ridgefill_s1_window;
            require(sweep->s1_probe_kv_pairs >= width * uint64_t(n_prefix) + width * (width + 1) / 2,
                    "S1 probe needs the dense query/key pair count from the completed driver");
            report.s1_probe_kv_pairs = sweep->s1_probe_kv_pairs;
        } else { require(sweep->s1_begin == -1 && sweep->s1_end == -1 && sweep->s1_probe_kv_pairs == 0, "unexpected S1 work"); }
        if (vtail) {
            require(sweep->vtail_rows == std::min(params.ridgefill_vtail_rows, n_prompt - n_prefix), "invalid completed vtail row count");
            report.vtail_rows = sweep->vtail_rows;
        } else { require(sweep->vtail_rows == 0, "unexpected vtail work"); }
    } else { require(!sweep || (sweep->s1_begin == -1 && sweep->s1_end == -1 && sweep->s1_probe_kv_pairs == 0 && sweep->vtail_rows == 0), "unexpected sweep work"); }
    double exact_keys = 0;
    for (auto row : rows) { exact_keys += double(row) + 1; }
    if (s1) {
        for (int32_t i = 0; i < n_prefix; ++i) {
            const bool selected = std::binary_search(rows.begin(), rows.end(), i);
            const bool special = params.ridgefill_special_exact && llama_ridgefill_token_is_special(rf, tokens[i]);
            require(!special || selected, "S1 special rows differ from the reported exact rows");
            if (!selected) { continue; }
            if (i < filled_begin || i >= filled_end) { ++report.n_cached_exact_prefix; }
            else { ++report.exact_by_source[special ? 2 : 5]; }
        }
    } else if (n_prefix > 0 && (mode == "quality" || mode == "quality+qattn")) {
        require(rf && tokens, "quality breakdown needs the projector and prompt tokens");
        std::vector<llama_pos> classes(n_prefix);
        const int32_t count = llama_ridgefill_select_rows(rf, tokens, n_prefix, params.ridgefill_share, false, classes.data());
        require(count >= 0 && count <= n_prefix, "class selector refused the breakdown");
        std::vector<uint8_t> source(n_prefix, 0);
        for (int32_t i = 0; i < count; ++i) {
            require(classes[i] >= 0 && classes[i] < n_prefix, "class selector returned an invalid position");
            source[classes[i]] = 1;
        }
        const auto * vocab = llama_model_get_vocab(model);
        const int excluded = LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_UNUSED;
        for (int32_t i = 0; i < n_prefix; ++i) {
            if (!source[i] && params.ridgefill_digits && !(llama_vocab_get_attr(vocab, tokens[i]) & excluded) &&
                    std::strpbrk(llama_vocab_get_text(vocab, tokens[i]), "0123456789")) { source[i] = 2; }
        }
        if (params.ridgefill_special_exact) {
            for (int32_t i = 0; i < n_prefix; ++i) {
                if (!source[i] && llama_ridgefill_token_is_special(rf, tokens[i])) { source[i] = 3; }
            }
        }
        int32_t header_end = 0;
        for (int32_t i = 0; i < n_prefix; ++i) {
            if (!source[i] && i < header_end) { source[i] = 4; }
            if (llama_ridgefill_token_is_special(rf, tokens[i])) {
                header_end = i + 1 + std::min(params.ridgefill_header_rows, n_prefix - i - 1);
            }
        }
        size_t next = 0;
        for (int32_t i = 0; i < n_prefix; ++i) {
            const bool selected = next < rows.size() && rows[next] == i;
            require(!source[i] || selected, "base selector rows differ from the reported exact rows");
            if (selected) {
                if (source[i]) {
                    ++report.exact_by_source[source[i] - 1];
                } else if (mode == "quality+qattn" && window_begin >= 0 && std::binary_search(selected_rows.begin(), selected_rows.end(), i)) {
                    ++report.exact_by_source[4];
                } else {
                    require(i < filled_begin || i >= filled_end, "unattributed exact prefix row");
                    ++report.n_cached_exact_prefix;
                }
                ++next;
            }
        }
    } else {
        require(selected_rows.empty(), "speed/off cannot select exact prefix rows");
        report.n_cached_exact_prefix = rows.size();
    }
    if (n_prompt == 0) { return report; }
    require(model->arch == LLM_ARCH_LLAMA || model->arch == LLM_ARCH_QWEN3 || model->arch == LLM_ARCH_QWEN35,
            "analytic MACs support llama, qwen3 and qwen35 only");
    const auto & hp = model->hparams;
    const uint32_t layers = hp.n_layer();
    const uint32_t split = rf ? llama_ridgefill_split(rf) : layers;
    require(split <= layers, "invalid split for analytic MACs");
    const double hidden = hp.n_embd;
    const double n = n_prompt, p = n_prefix, r = rows.size(), filled = p - r;
    const double head = hidden * llama_vocab_n_tokens(llama_model_get_vocab(model));
    report.macs_exact = report.macs_ridgefill = head;
    require((window_begin == -1 && window_end == -1) ||
            (window_begin >= 0 && window_end > window_begin && window_end <= n_prompt), "invalid analytic query window");
    for (uint32_t il = 0; il < layers; ++il) {
        const double mlp = 3 * hidden * hp.n_ff(il);
        double linear, fill, context = 0, tail_context = 0, row_context = 0;
        if (hp.is_recr(il)) {
            require(hp.ssm_dt_rank > 0 && hp.ssm_d_inner % hp.ssm_dt_rank == 0, "invalid GDN value heads");
            const double nk = hp.ssm_n_group, nv = hp.ssm_dt_rank, dk = hp.ssm_d_state;
            const double dv = hp.ssm_d_inner / hp.ssm_dt_rank;
            const double conv = 2 * nk * dk + nv * dv;
            const double inproj = hidden * conv + 2 * hidden * nv + conv * hp.ssm_d_conv;
            linear = inproj + 2 * hidden * nv * dv + 4 * nv * dk * dv;
            fill = inproj + 3 * nv * dk * dv;
        } else {
            require(hp.n_embd_head_k(il) == hp.n_embd_head_v(il), "analytic attention requires equal key/value head widths");
            const double hq = hp.n_head(il), hkv = hp.n_head_kv(il), dim = hp.n_embd_head_k(il);
            const double query = hidden * hq * dim * (model->arch == LLM_ARCH_QWEN35 ? 2 : 1);
            fill = 2 * hidden * hkv * dim;
            linear = query + fill + hidden * hq * dim;
            context = hq * dim * n * (n + 1);
            tail_context = hq * dim * (n - p) * (p + n + 1);
            row_context = 2 * hq * dim * exact_keys;
            if (window_begin >= 0 && il < split && (il == 3 || il == 7 || il == 11 || il == 15)) {
                const double window = window_end - window_begin;
                report.macs_ridgefill += window * query + hq * dim * window * (double(window_begin) + window_end + 1) / 2;
            }
        }
        if (il >= split) {
            if (s1) {
                const double window = report.s1_end - report.s1_begin;
                report.macs_s1_probe += window * (linear + mlp);
                if (!hp.is_recr(il)) {
                    report.macs_s1_fill += p * (hidden * hidden + fill);
                    report.macs_s1_probe += 2.0 * hp.n_head(il) * hp.n_embd_head_k(il) * report.s1_probe_kv_pairs;
                }
            }
            if (vtail && !hp.is_recr(il)) {
                report.macs_vtail += report.vtail_rows * (hidden * hidden + fill);
            }
        }
        const double exact = n * (linear + mlp) + context;
        report.macs_exact += exact;
        report.macs_ridgefill += il < split ? exact :
            (n - p) * (linear + mlp) + tail_context + filled * (fill + hidden * hidden) + r * (linear + mlp) + row_context;
    }
    report.macs_ridgefill += report.macs_s1_fill + report.macs_s1_probe + report.macs_vtail;
    require(std::isfinite(report.macs_exact) && std::isfinite(report.macs_ridgefill) && report.macs_exact > 0,
            "nonfinite analytic MAC count");
    return report;
}

void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora) {
    std::vector<llama_adapter_lora *> loras;
    std::vector<float> scales;

    for (auto & la: lora) {
        loras.push_back(la.ptr);
        scales.push_back(la.scale);
    }

    llama_set_adapters_lora(ctx, loras.data(), loras.size(), scales.data());
}

struct llama_model_params common_model_params_to_llama(common_params & params) {
    auto mparams = llama_model_default_params();

    if (!params.devices.empty()) {
        mparams.devices = params.devices.data();
    }

    mparams.n_gpu_layers    = params.n_gpu_layers;
    mparams.main_gpu        = params.main_gpu;
    mparams.split_mode      = params.split_mode;
    mparams.load_mode       = params.load_mode;
    mparams.lazy_mode = params.lazy_mode;
    mparams.tensor_split    = params.tensor_split;
    mparams.check_tensors   = params.check_tensors;
    mparams.use_extra_bufts = !params.no_extra_bufts;
    mparams.no_host         = params.no_host;

    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    if (params.tensor_buft_overrides.empty()) {
        mparams.tensor_buft_overrides = NULL;
    } else {
        GGML_ASSERT(params.tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern");
        mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    }

    mparams.progress_callback           = params.load_progress_callback;
    mparams.progress_callback_user_data = params.load_progress_callback_user_data;
    mparams.no_alloc                    = params.no_alloc;
    mparams.load_mtp                    = params.load_mtp || std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    return mparams;
}

struct llama_context_params common_context_params_to_llama(const common_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = params.n_ctx;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_rs_seq          = params.speculative.need_n_rs_seq();
    cparams.n_outputs_max     = std::max(params.n_outputs_max, 0);
    cparams.n_outputs_max_per_seq = std::max(params.n_outputs_max_per_seq, 0);
    cparams.n_batch           = params.n_batch;
    cparams.n_ubatch          = params.n_ubatch;
    cparams.n_threads         = params.cpuparams.n_threads;
    cparams.n_threads_batch   = params.cpuparams_batch.n_threads == -1 ?
                                params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    cparams.embeddings        = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.pooling_type      = params.pooling_type;
    cparams.attention_type    = params.attention_type;
    cparams.flash_attn_type   = params.flash_attn_type;
    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.no_perf           = params.no_perf;
    cparams.op_offload        = !params.no_op_offload;
    cparams.swa_full          = params.swa_full;
    cparams.kv_unified        = params.kv_unified;

    cparams.type_k = params.cache_type_k;
    cparams.type_v = params.cache_type_v;

    cparams.moe_cache_size = params.moe_cache_size;

    return cparams;
}

//
// Threadpool utils
//

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = params.priority;
    tpp.poll       = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

common_threadpools::~common_threadpools() {
    if (!free_fn) {
        return;
    }
    free_fn(threadpool);
    free_fn(threadpool_batch);
}

void common_threadpools::init(llama_context * ctx, const common_params & params) {
    GGML_ASSERT(!threadpool);
    GGML_ASSERT(!threadpool_batch);

    COM_INF("llama threadpool init, n_threads = %d\n", (int) params.cpuparams.n_threads);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        COM_WRN("%s", "no CPU backend found\n");
        return;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    // each pool needs to match the respective n_threads exactly
    // see: https://github.com/ggml-org/llama.cpp/pull/27138#issuecomment-5332307332
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            COM_WRN("batch threadpool create failed : n_threads %d\n", tpp_batch.n_threads);
            return;
        }

        // start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        COM_WRN("threadpool create failed : n_threads %d\n", tpp.n_threads);
        free_fn(threadpool_batch);
        threadpool_batch = nullptr;
        return;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);
}

//
// Vocab utils
//

std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string common_detokenize(const struct llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_detokenize(vocab, tokens, special);
}

std::string common_detokenize(const struct llama_vocab * vocab, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//
// Embedding utils
//

void common_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n){
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum  += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0) {
        if (sum1 == 0.0 && sum2 == 0.0) {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

//
// Control vector utils
//

static common_control_vector_data common_control_vector_load_one(const common_control_vector_load_info & load_info) {
    common_control_vector_data result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(load_info.fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        COM_ERR("failed to load control vector file from %s\n", load_info.fname.c_str());
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        COM_WRN("no direction tensors found in %s\n", load_info.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            COM_ERR("invalid/unparsable direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            COM_ERR("invalid (zero) direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            COM_ERR("invalid (non-F32) direction tensor type in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            COM_ERR("invalid (non-1D) direction tensor shape in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            COM_ERR("direction tensor in %s does not match previous dimensions\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.n_embd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.n_embd == -1) {
        COM_WRN("skipping %s due to invalid direction tensors\n", load_info.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos) {
    common_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = common_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            COM_ERR("control vectors in %s does not match previous dimensions\n", info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        COM_ERR("%s", "no valid control vector files passed\n");
        result.data.clear();
    }

    return result;
}

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    const int64_t ndata        = (tokens.size() - ne_datapoint - 1) / stride;
    ggml_opt_dataset_t result = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, ne_datapoint, ne_datapoint, ndata, /*ndata_shard =*/ 1);

    llama_token * data   = (llama_token *) ggml_opt_dataset_data(result)->data;
    llama_token * labels = (llama_token *) ggml_opt_dataset_labels(result)->data;

    for (int64_t idata = 0; idata < ndata; ++idata) {
        memcpy(data   + idata*ne_datapoint, tokens.data() + idata*stride + 0, ne_datapoint*sizeof(llama_token));
        memcpy(labels + idata*ne_datapoint, tokens.data() + idata*stride + 1, ne_datapoint*sizeof(llama_token));
    }

    return result;
}

ggml_opt_optimizer_params common_opt_lr_pars(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    const lr_opt &            d      = *(lr_opt *) userdata;
    result.adamw.alpha = result.sgd.alpha = d.get_lr(d.epoch);
    result.sgd.wd = result.adamw.wd = d.wd;
    return result;
}

// TODO make all command line args case-insensitive
static inline bool eq_case_insensitive(char const* a, char const* b) {
    return !
#if defined(_MSC_VER)
        _stricmp
#else
        strcasecmp
#endif // defined(_MSC_VER)
        (a, b);
}

enum ggml_opt_optimizer_type common_opt_get_optimizer(const char * n) {
    if (eq_case_insensitive("adamw", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    }
    if (eq_case_insensitive("sgd", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_SGD;
    }
    return GGML_OPT_OPTIMIZER_TYPE_COUNT;
}

// TODO simplify to use just log and exp
static float const k_log_2 = std::log(2.f);

void lr_opt::init() {
    if (lr_min > 0 && lr_min < lr0) {
        float nhalf = std::log(lr0 / lr_min) / k_log_2;
        float e     = epochs;
        if (decay_epochs > 0 && decay_epochs < e) {
            e = decay_epochs;
        } else {
            decay_epochs = e;
        }
        scale_epoch = nhalf / e;
    }
}

float lr_opt::get_lr(float epoch) const {
    float r = lr_min <= 0 ? lr0 :
        epoch >= decay_epochs ? lr_min :
        lr0 * std::pow(0.5f, epoch * scale_epoch);
    LOG_INF("epoch %.2g lr=%.2g\n", epoch, r);
    return r;
}

bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos) {
    common_batch batch(ctx);
    batch.add(last_token, pos, 0, true);

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("%s: failed to replay last token\n", __func__);
        return false;
    }
    return true;
}

common_batch::common_batch(llama_context * ctx) : batch(llama_batch_ext_init(ctx)) {
    const auto rope_type = llama_model_rope_type(llama_get_model(ctx));
    n_pos = rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? GGML_MROPE_SECTIONS : 1;
}

void common_batch::clear() {
    tokens.clear();
}

int32_t common_batch::add(llama_token id, llama_pos pos, llama_seq_id seq_id, bool output) {
    tokens.push_back({ id, { pos, 0, 0, 0 }, seq_id, output, { nullptr, 0, 0 }, { nullptr, 0, 0 }, {} });
    return size() - 1;
}

int32_t common_batch::add(llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool output) {
    GGML_ASSERT(!seq_ids.empty());

    const int32_t idx = add(id, pos, seq_ids[0], output);
    for (size_t s = 1; s < seq_ids.size(); ++s) {
        add_seq(idx, seq_ids[s]);
    }
    return idx;
}

bool common_batch::add_seq(int32_t idx, llama_seq_id seq_id) {
    if (idx < 0 || idx >= size()) {
        return false;
    }
    tokens[idx].seq_ids_extra.push_back(seq_id);
    return true;
}

bool common_batch::set_output(int32_t idx, bool value) {
    if (idx < 0 || idx >= size()) {
        return false;
    }
    tokens[idx].output = value;
    return true;
}

bool common_batch::set_embd(int32_t idx, llama_embd embd) {
    if (idx < 0 || idx >= size() || tokens[idx].embd.data != nullptr) {
        return false;
    }
    tokens[idx].embd = embd;
    return true;
}

bool common_batch::set_embd_state(int32_t idx, llama_embd state) {
    if (idx < 0 || idx >= size() || tokens[idx].state.data != nullptr) {
        return false;
    }
    tokens[idx].state = state;
    return true;
}

int32_t common_batch::add_embd(llama_embd embd, const llama_pos * pos, llama_seq_id seq_id, bool output) {
    token t = { LLAMA_TOKEN_NULL, { 0, 0, 0, 0 }, seq_id, output, embd, { nullptr, 0, 0 }, {} };
    for (int32_t j = 0; j < n_pos; ++j) {
        t.pos[j] = pos[j];
    }
    tokens.push_back(t);
    return size() - 1;
}

llama_batch_ext * common_batch::get_sub_batch(int32_t off, int32_t n) {
    GGML_ASSERT(batch && "common_batch was not initialized with a context");
    GGML_ASSERT(off >= 0 && n >= 0 && off + n <= size());

    llama_batch_ext * res = batch.get();
    llama_batch_ext_clear(res);

    for (int32_t i = off; i < off + n; ++i) {
        const token & t = tokens[i];

        int32_t idx;
        if (t.id != LLAMA_TOKEN_NULL) {
            idx = llama_batch_ext_add_token(res, t.seq_id, t.id);
            if (idx < 0) {
                GGML_ABORT("%s: failed to add token %d at index %d (error %d, n = %d)\n", __func__, t.id, i, idx, n);
            }
            llama_batch_ext_set_pos(res, idx, t.pos.data());
            if (t.embd.data && !llama_batch_ext_set_embd_token(res, idx, t.embd)) {
                GGML_ABORT("%s: failed to set the embedding of token %d at index %d\n", __func__, t.id, i);
            }
        } else {
            idx = llama_batch_ext_add_embd(res, t.seq_id, t.embd);
            if (idx < 0) {
                GGML_ABORT("%s: failed to add embedding at index %d (error %d, n = %d)\n", __func__, i, idx, n);
            }
            llama_batch_ext_set_pos(res, idx, t.pos.data());
        }
        GGML_ASSERT(idx == i - off);

        for (const llama_seq_id seq_id : t.seq_ids_extra) {
            if (!llama_batch_ext_add_seq(res, idx, seq_id)) {
                GGML_ABORT("%s: failed to add seq %d to the entry at index %d\n", __func__, seq_id, i);
            }
        }
        if (t.output) {
            llama_batch_ext_set_output_logits(res, idx, true);
        }
        if (t.state.data) {
            llama_batch_ext_set_embd_state(res, idx, t.state); // contexts without a state input ignore it
        }
        if (t.decision_order != 0) {
            llama_batch_ext_set_decision_order(res, idx, (llama_decision_order) t.decision_order);
        }
    }

    return res;
}

common_batch common_batch_get_one(llama_context * ctx, const llama_token * tokens, int32_t n_tokens) {
    common_batch batch(ctx);

    auto mem = llama_get_memory(ctx);
    llama_pos pos = llama_memory_seq_pos_max(mem, 0) + 1; // -1 + 1 == 0 when the memory is empty

    for (int32_t i = 0; i < n_tokens; ++i) {
        const bool output = i == n_tokens - 1;
        batch.add(tokens[i], pos, 0, output);
        pos++;
    }

    return batch;
}

common_batch common_batch_get_one(llama_context * ctx, const llama_tokens & tokens) {
    return common_batch_get_one(ctx, tokens.data(), (int32_t) tokens.size());
}

bool common_prompt_batch_decode(
              struct llama_context * ctx,
                const llama_tokens & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state) {
    if (n_new == 0) {
        return true;
    }
    const int offset = all_tokens.size() - n_new;

    if (save_state && n_new > 1) {
        const int n_tokens_before_last = n_new - 1;

        GGML_ASSERT(n_new <= n_batch);

        // Decode all but the last token so we can save the memory state before decoding the last token.
        // This is done so we can restore the session state later and replay the last token.
        // Memory implementations in recurrent/hybrid models don't support removing tokens from their
        // memory, so we can't just remove the last token from the memory and replay the last token which
        // is the reason for this logic.
        llama_tokens prefix_tokens(all_tokens.begin() + offset, all_tokens.begin() + offset + n_tokens_before_last);
        common_batch batch_prefix = common_batch_get_one(ctx, prefix_tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch_prefix.get())) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_tokens_before_last;

        llama_state_save_file(ctx, state_path.data(), all_tokens.data(), all_tokens.size());
        COM_INF("saved session before last token to %s, n_new = %zu\n", state_path.data(), all_tokens.size());

        common_batch batch_last(ctx);
        batch_last.add(all_tokens.back(), n_past, 0, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch_last.get())) {
            COM_ERR("%s", "failed to eval last token\n");
            return false;
        }
        n_past++;
    } else {
        llama_tokens new_tokens(all_tokens.begin() + offset, all_tokens.begin() + offset + n_new);
        common_batch batch = common_batch_get_one(ctx, new_tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_new;
    }

    return true;
}

size_t common_prompt_checkpoint::size() const {
    return data_tgt.size() + data_dft.size() + data_spec.size();
}

bool common_prompt_checkpoint::empty() const {
    return data_tgt.empty();
}

void common_prompt_checkpoint::clear() {
    n_tokens = 0;

    pos_min = 0;
    pos_max = 0;

    data_tgt.clear();
    data_dft.clear();
    data_spec.clear();
}

void common_prompt_checkpoint::update_pos(
        int64_t n_tokens,
        llama_pos pos_min,
        llama_pos pos_max) {
    this->n_tokens = n_tokens;
    this->pos_min  = pos_min;
    this->pos_max  = pos_max;
}

void common_prompt_checkpoint::update_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    data_tgt.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_tgt.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

void common_prompt_checkpoint::update_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    data_dft.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_dft.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

bool common_prompt_checkpoint::load_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return true;
    }

    if (data_tgt.empty()) {
        return true;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_tgt.data(), data_tgt.size(), seq_id, flags);
    return n == data_tgt.size();
}

bool common_prompt_checkpoint::load_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return true;
    }

    if (data_dft.empty()) {
        return true;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_dft.data(), data_dft.size(), seq_id, flags);
    return n == data_dft.size();
}

void common_prompt_checkpoint::clear_tgt() {
    data_tgt.clear();
}

void common_prompt_checkpoint::clear_dft() {
    data_dft.clear();
    data_spec.clear();
}
