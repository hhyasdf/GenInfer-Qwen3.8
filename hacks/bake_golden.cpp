// bake_golden.cpp — Independent golden baker for the gen-inference engine.
//
// Loads the Q6_K GGUF via the llama.cpp C API (a code path fully independent
// of the Python engine), and for each fixed prompt:
//   (a) tokenizes + evals the prompt,
//   (b) dumps the FULL last-position logits (llama_get_logits_ith),
//   (c) greedy-decodes N tokens (pure argmax, no temperature),
//   (d) writes golden/logits.json in the gen-inference-golden/v1 format.
//
// The golden's `model` field is the GGUF *directory* (the engine's
// args.model_path), NOT the .gguf file — the engine canonicalizes a directory
// to its absolute path, and the selftest compares `model` against
// canonical_model_id(args.model_path).
//
// Build:
//   g++ -O2 -std=c++17 \
//       -I/home/hhy/Projects/llama.cpp/include \
//       -I/home/hhy/Projects/llama.cpp/ggml/include \
//       bake_golden.cpp -o bake_golden \
//       -L/home/hhy/Projects/llama.cpp/build/bin -l:libllama.so.0 \
//       -l:libggml.so.0 \
//       -Wl,-rpath,/home/hhy/Projects/llama.cpp/build/bin
//
// Run (as a gpu-job, since the origin holds the GPU):
//   ./bake_golden --model <gguf-dir> --out golden/logits.json \
//       --decode-steps 30 --tolerance 0.2

#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// JSON helpers (no external deps)
// ---------------------------------------------------------------------------

static void json_escape(const std::string &in, std::string &out) {
    out.clear();
    for (unsigned char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    std::string model_dir;   // GGUF directory (the engine's args.model_path)
    std::string model_file;  // the .gguf file inside model_dir
    std::string out;         // golden/logits.json
    int decode_steps = 30;   // greedy tokens to bake (>= 30 per the skill)
    float tolerance = 0.2f;  // max_abs_diff on logits (measured, with margin)
    int greedy_budget = 3;   // greedy_max_mismatches (10% of 30)
    int n_ctx = 8192;        // context for the reference (prompts are short)
    int n_batch = 512;
    int main_gpu = 0;
    bool verbose = true;
};

static bool parse_args(int argc, char **argv, Args &a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--model") {
            a.model_dir = next("--model");
        } else if (k == "--model-file") {
            a.model_file = next("--model-file");
        } else if (k == "--out") {
            a.out = next("--out");
        } else if (k == "--decode-steps") {
            a.decode_steps = std::atoi(next("--decode-steps").c_str());
        } else if (k == "--tolerance") {
            a.tolerance = std::atof(next("--tolerance").c_str());
        } else if (k == "--greedy-budget") {
            a.greedy_budget = std::atoi(next("--greedy-budget").c_str());
        } else if (k == "--n-ctx") {
            a.n_ctx = std::atoi(next("--n-ctx").c_str());
        } else if (k == "--n-batch") {
            a.n_batch = std::atoi(next("--n-batch").c_str());
        } else if (k == "--main-gpu") {
            a.main_gpu = std::atoi(next("--main-gpu").c_str());
        } else if (k == "--quiet") {
            a.verbose = false;
        } else if (k == "--help" || k == "-h") {
            std::printf(
                "usage: bake_golden --model <gguf-dir> --out <json> "
                "[--model-file <gguf>] [--decode-steps N] "
                "[--tolerance F] [--greedy-budget N] [--n-ctx N] "
                "[--n-batch N] [--main-gpu N] [--quiet]\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", k.c_str());
            std::exit(2);
        }
    }
    if (a.model_dir.empty() || a.out.empty()) {
        std::fprintf(stderr, "--model and --out are required\n");
        std::exit(2);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Prompts (fixed, >= 50 tokens each, per the skill's golden requirements)
// ---------------------------------------------------------------------------

static std::vector<std::string> default_prompts() {
    return {
        "Explain in detail the difference between a stack and a queue in "
        "data structures, including the time complexity of each operation "
        "(push, pop, enqueue, dequeue) and provide concrete examples of "
        "when you would use each one in a real-world application. Also "
        "discuss the space complexity of each data structure and how they "
        "are implemented in common programming languages such as Python "
        "and C++.",
        "Write a Python function that takes a list of integers as input and "
        "returns the sum of all even numbers in the list. The function "
        "should handle edge cases such as an empty list, a list with no "
        "even numbers, and a list with negative numbers. Include docstrings "
        "and type hints in your answer.",
        "Once upon a time, in a land far away, there lived a young dragon "
        "who dreamed of flying across the vast ocean to find his lost "
        "family. Every night, he would look up at the stars and wonder if "
        "his parents were still alive somewhere in the world. One day, he "
        "decided to embark on a journey to find them, no matter how "
        "dangerous the path might be.",
    };
}

// ---------------------------------------------------------------------------
// Batch helpers (this llama.cpp version has no llama_batch_add; set fields)
// ---------------------------------------------------------------------------

// Build a single-sequence batch over [tokens, tokens+n) at positions
// [start_pos, start_pos+n). Only the last token requests output logits.
static llama_batch make_batch(const llama_token *tokens, int32_t n,
                              llama_pos start_pos) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = start_pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == n - 1) ? 1 : 0;
    }
    batch.n_tokens = n;
    return batch;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    Args args;
    parse_args(argc, argv, args);

    if (args.model_file.empty()) {
        // Default: the main model is the Q6_K file in the directory.
        args.model_file =
            args.model_dir + "/Qwen3.8-27B-EfficientThink-SimPO-Q6_K.gguf";
    }

    if (args.verbose) {
        std::printf("bake_golden: model=%s\n", args.model_file.c_str());
        std::printf("bake_golden: out=%s decode_steps=%d tolerance=%.4f\n",
                    args.out.c_str(), args.decode_steps, args.tolerance);
    }

    // Load all backends (CUDA + CPU) so the model can be split across GPUs.
    ggml_backend_load_all();

    // --- Load the model (all layers on GPU, tensor-split across both GPUs,
    // matching the origin's --device CUDA0,CUDA1 --tensor-split 22,22) ---
    static const float tensor_split[2] = {0.5f, 0.5f};
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;  // all layers on GPU
    mparams.split_mode = LLAMA_SPLIT_MODE_TENSOR;  // 50/50 tensor split
    mparams.tensor_split = tensor_split;
    mparams.main_gpu = args.main_gpu;

    llama_model *model = llama_model_load_from_file(args.model_file.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", args.model_file.c_str());
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (args.verbose) {
        std::printf("bake_golden: vocab=%d\n", n_vocab);
    }

    // --- Init the context ---
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = args.n_ctx;
    cparams.n_batch = args.n_batch;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    cparams.no_perf = true;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to init context\n");
        return 1;
    }

    // --- Greedy sampler (pure argmax, no temperature) ---
    llama_sampler *smpl =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // --- Prompts ---
    std::vector<std::string> prompts = default_prompts();

    // --- Open the output file ---
    std::ofstream out(args.out.c_str());
    if (!out.is_open()) {
        std::fprintf(stderr, "failed to open output: %s\n", args.out.c_str());
        return 1;
    }

    // --- Write the JSON header ---
    out << "{\n";
    out << "  \"format\": \"gen-inference-golden/v1\",\n";
    out << "  \"model\": \"";
    std::string esc_model;
    json_escape(args.model_dir, esc_model);
    out << esc_model << "\",\n";
    out << "  \"dtype\": \"bf16\",\n";
    out << "  \"tolerance\": { \"max_abs_diff\": " << args.tolerance << " },\n";
    out << "  \"prompts\": [\n";

    // --- Process each prompt ---
    for (size_t p = 0; p < prompts.size(); ++p) {
        const std::string &text = prompts[p];
        if (args.verbose) {
            std::printf("bake_golden: prompt %zu/%zu (%zu chars)\n",
                        p + 1, prompts.size(), text.size());
        }

        // Tokenize the prompt (add_bos on, special off).
        std::vector<llama_token> ids;
        {
            int n = llama_tokenize(
                vocab, text.c_str(), (int)text.size(),
                ids.data(), (int)ids.capacity(), true /* add_bos */,
                false /* special */);
            if (n < 0) {
                ids.resize(-n);
                n = llama_tokenize(
                    vocab, text.c_str(), (int)text.size(),
                    ids.data(), (int)ids.size(), true, false);
            }
            if (n < 0) {
                std::fprintf(stderr, "tokenize failed for prompt %zu\n", p);
                return 1;
            }
            ids.resize(n);
        }

        if (args.verbose) {
            std::printf("bake_golden:   %zu tokens\n", ids.size());
        }

        // Reset the KV cache for this prompt.
        llama_memory_clear(llama_get_memory(ctx), true);

        // Eval the prompt in chunks of n_batch (prompts are short, one chunk).
        {
            size_t i = 0;
            while (i < ids.size()) {
                size_t chunk = std::min<size_t>(args.n_batch, ids.size() - i);
                llama_batch batch =
                    make_batch(ids.data() + i, (int32_t)chunk, (llama_pos)i);
                if (llama_decode(ctx, batch) != 0) {
                    std::fprintf(stderr, "decode failed for prompt %zu\n", p);
                    return 1;
                }
                llama_batch_free(batch);
                i += chunk;
            }
        }

        // Dump the FULL last-position logits.
        //
        // Two hazards make holding the returned pointer unsafe:
        //   (1) llama_decode copies the logits to the host buffer
        //       ASYNCHRONOUSLY (cudaMemcpyAsync on the context stream) and
        //       returns before the copy lands — synchronize before reading.
        //   (2) the host buffer is REUSED by every subsequent decode, so the
        //       rows must be copied out NOW, before the greedy loop below.
        const int32_t last_pos = (int32_t)ids.size() - 1;
        llama_synchronize(ctx);
        const float *logits = llama_get_logits_ith(ctx, last_pos);
        if (!logits) {
            std::fprintf(stderr, "failed to get logits for prompt %zu\n", p);
            return 1;
        }
        std::vector<float> prefill_logits(logits, logits + n_vocab);

        // Greedy decode for N steps.
        std::vector<llama_token> greedy_ids;
        greedy_ids.reserve(args.decode_steps);
        for (int step = 0; step < args.decode_steps; ++step) {
            // Sample from the current logits (last output position). The
            // preceding decode's async logits copy is complete (synced).
            llama_token next = llama_sampler_sample(smpl, ctx, -1);
            greedy_ids.push_back(next);
            llama_sampler_reset(smpl);

            // Eval the sampled token (single-token batch, output logits on).
            llama_token tok[1] = {next};
            llama_batch dbatch =
                make_batch(tok, 1, (llama_pos)(last_pos + 1 + step));
            if (llama_decode(ctx, dbatch) != 0) {
                std::fprintf(stderr, "decode failed at step %d\n", step);
                return 1;
            }
            llama_batch_free(dbatch);
            // Ensure the async logits copy has landed before the next
            // sample reads the host buffer (decode returns before the
            // cudaMemcpyAsync completes).
            llama_synchronize(ctx);
        }

        // Write the prompt's JSON entry.
        if (p > 0) out << ",\n";
        out << "    {\n";
        out << "      \"text\": \"";
        std::string esc_text;
        json_escape(text, esc_text);
        out << esc_text << "\",\n";

        // ids
        out << "      \"ids\": [";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) out << ", ";
            out << ids[i];
        }
        out << "],\n";

        // logits (full vocab, full precision) — the prefill rows copied out
        // before the greedy loop (the context buffer is reused by then).
        out << "      \"logits\": [";
        out.precision(17);
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (i) out << ", ";
            out << prefill_logits[i];
        }
        out << "],\n";

        // greedy_ids
        out << "      \"greedy_ids\": [";
        for (size_t i = 0; i < greedy_ids.size(); ++i) {
            if (i) out << ", ";
            out << greedy_ids[i];
        }
        out << "],\n";

        // greedy_max_mismatches
        out << "      \"greedy_max_mismatches\": " << args.greedy_budget << "\n";
        out << "    }";

        if (args.verbose) {
            std::printf("bake_golden:   baked %zu greedy tokens\n",
                        greedy_ids.size());
        }
    }

    // --- Write the JSON footer ---
    out << "\n  ]\n";
    out << "}\n";
    out.close();

    if (args.verbose) {
        std::printf("bake_golden: wrote %s\n", args.out.c_str());
    }

    // --- Cleanup ---
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
