// Shannon-Prime Engine — HTTP server impl.
// See http_server.h for context.

#include "http_server.h"

#include "httplib.h"   // single-header, vendored at vendor/cpp-httplib/

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace sp::engine {

namespace {

// ──────────────────────────────────────────────────────────────────────
// Minimal JSON helpers — our request/response shape is tiny and known,
// so we avoid pulling in a full JSON library. Robust enough for the
// proxy's output shape (which we control on the response side) and the
// proxy's request shape (which we parse defensively).
// ──────────────────────────────────────────────────────────────────────

// Find the JSON string value for `"key":` in `s`. Returns the unescaped
// content of the string (without surrounding quotes). Empty string if
// not found or malformed. Handles \" and \\ escapes.
static std::string json_get_string(const std::string& s, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while (true) {
        pos = s.find(needle, pos);
        if (pos == std::string::npos) return "";
        // After the key we expect optional whitespace then a colon.
        size_t p = pos + needle.size();
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) ++p;
        if (p >= s.size() || s[p] != ':') { pos += needle.size(); continue; }
        ++p;
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) ++p;
        if (p >= s.size() || s[p] != '"') return "";  // not a string value
        // Read string with \-escape handling.
        ++p;
        std::string out;
        while (p < s.size()) {
            char c = s[p];
            if (c == '\\' && p + 1 < s.size()) {
                char e = s[p + 1];
                switch (e) {
                    case '"':  out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    default:   out.push_back(e); break;  // u-escape ignored for now
                }
                p += 2;
            } else if (c == '"') {
                return out;
            } else {
                out.push_back(c);
                ++p;
            }
        }
        return out;  // unterminated string — best-effort
    }
}

// Find the JSON integer value for `"key":N`. Returns `dflt` if not found.
static int json_get_int(const std::string& s, const std::string& key, int dflt) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = s.find(needle);
    if (pos == std::string::npos) return dflt;
    size_t p = pos + needle.size();
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == ':')) ++p;
    if (p >= s.size()) return dflt;
    bool neg = false;
    if (s[p] == '-') { neg = true; ++p; }
    if (p >= s.size() || s[p] < '0' || s[p] > '9') return dflt;
    int v = 0;
    while (p < s.size() && s[p] >= '0' && s[p] <= '9') {
        v = v * 10 + (s[p] - '0');
        ++p;
    }
    return neg ? -v : v;
}

// Escape a string for embedding in JSON output.
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    return out;
}

// Extract the chat messages array as parallel role/content vectors.
// We scan for {"role":"...","content":"..."} occurrences inside the
// "messages": [ ... ] block. Order-preserving.
static void extract_chat_messages(const std::string& body,
                                  std::vector<std::string>& roles,
                                  std::vector<std::string>& contents) {
    size_t arr_start = body.find("\"messages\"");
    if (arr_start == std::string::npos) return;
    arr_start = body.find('[', arr_start);
    if (arr_start == std::string::npos) return;
    size_t arr_end = body.find(']', arr_start);
    if (arr_end == std::string::npos) arr_end = body.size();

    // Walk objects {...} inside [arr_start, arr_end).
    size_t p = arr_start + 1;
    while (p < arr_end) {
        size_t obj_start = body.find('{', p);
        if (obj_start == std::string::npos || obj_start >= arr_end) break;
        // Find matching close brace, accounting for nested strings.
        int depth = 1;
        size_t q = obj_start + 1;
        bool in_str = false;
        while (q < body.size() && depth > 0) {
            char c = body[q];
            if (in_str) {
                if (c == '\\' && q + 1 < body.size()) { q += 2; continue; }
                if (c == '"') in_str = false;
            } else {
                if (c == '"') in_str = true;
                else if (c == '{') ++depth;
                else if (c == '}') --depth;
            }
            ++q;
        }
        const std::string obj = body.substr(obj_start, q - obj_start);
        const std::string role = json_get_string(obj, "role");
        const std::string content = json_get_string(obj, "content");
        if (!role.empty()) {
            roles.push_back(role);
            contents.push_back(content);
        }
        p = q;
    }
}

// Build a Qwen2/3-style ChatML prompt from messages. Hardcoded for now;
// follow-up reads chat_template from GGUF metadata via the tokenizer.
//
// Format:
//   <|im_start|>system
//   {sys}<|im_end|>
//   <|im_start|>user
//   {user}<|im_end|>
//   <|im_start|>assistant
static std::string build_chatml_prompt(const std::vector<std::string>& roles,
                                       const std::vector<std::string>& contents) {
    std::string out;
    for (size_t i = 0; i < roles.size(); ++i) {
        out += "<|im_start|>";
        out += roles[i];
        out += "\n";
        out += contents[i];
        out += "<|im_end|>\n";
    }
    out += "<|im_start|>assistant\n";
    return out;
}

// Generate a chatcmpl-XYZ id like OpenAI does. We don't need crypto-
// strength randomness; a thread-local PRNG is plenty.
static std::string gen_chatcmpl_id() {
    static thread_local std::mt19937_64 rng{
        std::random_device{}() ^ (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count()
    };
    static const char alpha[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string out = "chatcmpl-";
    for (int i = 0; i < 24; ++i) out.push_back(alpha[rng() % (sizeof(alpha) - 1)]);
    return out;
}

// Strip end-of-turn / special tokens from the model's output. We strip
// at the EARLIEST occurrence of any known stop token so the user never
// sees ChatML/EOS markers or model-emitted role-prefixes that come from
// the chat template "leaking" past the assistant's natural stop.
//
// Stops we catch:
//   <|im_end|>      — Qwen ChatML end-of-turn
//   <|endoftext|>   — generic EOS in many model vocabs
//   \nHuman:        — some models continue into a fake user turn
//   \n\nUser:       — same idea
//   \n\nuser:       — case variant
//
// Also catches partial-prefix at the very end of `s` (when generation
// truncates mid-token because n_predict ran out, e.g. "...today? <|im").
static std::string strip_chatml_tail(const std::string& s) {
    static const char* const stops[] = {
        "<|im_end|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "\nHuman:",
        "\n\nUser:",
        "\n\nuser:",
        nullptr
    };
    // Pass 1: find the earliest complete-stop occurrence; strip from there.
    std::string out = s;
    size_t earliest = std::string::npos;
    for (size_t i = 0; stops[i]; ++i) {
        const size_t pos = out.find(stops[i]);
        if (pos != std::string::npos && pos < earliest) earliest = pos;
    }
    if (earliest != std::string::npos) out = out.substr(0, earliest);

    // Pass 2: strip partial-prefix at the very end. The model may emit a
    // truncated special token (e.g. "<|im_" when n_predict ran out) just
    // before a different complete stop — pass 1 cuts at the complete stop
    // but leaves the partial behind. Walk longest-prefix-first so we
    // never under-strip.
    bool changed;
    do {
        changed = false;
        for (size_t i = 0; stops[i]; ++i) {
            const std::string stop = stops[i];
            for (size_t k = stop.size() - 1; k > 0; --k) {
                if (out.size() < k) continue;
                if (out.compare(out.size() - k, k, stop, 0, k) == 0) {
                    out = out.substr(0, out.size() - k);
                    changed = true;
                    break;
                }
            }
            if (changed) break;
        }
    } while (changed);
    return out;
}

} // anon

// ──────────────────────────────────────────────────────────────────────
// HttpServer implementation
// ──────────────────────────────────────────────────────────────────────

struct HttpServer::Impl {
    Engine*               engine = nullptr;
    std::string           model_name;
    std::string           web_root;
    httplib::Server       svr;
    std::atomic<bool>     running{false};
};

HttpServer::HttpServer() : impl_(std::make_unique<Impl>()) {}
HttpServer::~HttpServer() = default;

void HttpServer::bind(Engine* engine, const std::string& model_name, const std::string& web_root) {
    impl_->engine = engine;
    impl_->model_name = model_name;
    impl_->web_root = web_root;
}

int HttpServer::listen_and_serve(const std::string& host, int port) {
    if (!impl_->engine) {
        std::fprintf(stderr, "[sp-engine:http] no engine bound\n");
        return -1;
    }

    Engine* engine = impl_->engine;
    const std::string& model_name = impl_->model_name;
    const std::string& web_root   = impl_->web_root;

    // ── Static files ──────────────────────────────────────────────
    if (!web_root.empty()) {
        std::fprintf(stderr, "[sp-engine:http] serving static files from %s\n", web_root.c_str());
        impl_->svr.set_mount_point("/", web_root);
    }

    // ── CORS — allow any origin so the frontend can connect from
    //    file:// or a different dev-server port. ───────────────────
    impl_->svr.set_pre_routing_handler(
        [](const httplib::Request& req, httplib::Response& res) -> httplib::Server::HandlerResponse {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            if (req.method == "OPTIONS") {
                res.status = 204;
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });

    // ── Health ────────────────────────────────────────────────────
    impl_->svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // ── Models list ───────────────────────────────────────────────
    // Match llama.cpp server's shape: { "object":"list", "data":[ ... ] }
    impl_->svr.Get("/v1/models", [&model_name](const httplib::Request&, httplib::Response& res) {
        const auto created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::string body =
            "{\"object\":\"list\",\"data\":["
            "{\"id\":\"" + json_escape(model_name) + "\","
            "\"object\":\"model\","
            "\"created\":" + std::to_string(created) + ","
            "\"owned_by\":\"shannon-prime\"}"
            "]}";
        res.set_content(body, "application/json");
    });

    // ── Chat completions (non-streaming) ──────────────────────────
    impl_->svr.Post("/v1/chat/completions",
        [engine, &model_name](const httplib::Request& req, httplib::Response& res) {
            const std::string& body = req.body;

            // Extract messages.
            std::vector<std::string> roles, contents;
            extract_chat_messages(body, roles, contents);
            if (roles.empty()) {
                res.status = 400;
                res.set_content(
                    "{\"error\":{\"message\":\"missing or empty messages array\"}}",
                    "application/json");
                return;
            }

            // Build prompt + run inference.
            const std::string prompt = build_chatml_prompt(roles, contents);
            const int n_predict = json_get_int(body, "max_tokens", 256);

            std::string out;
            std::fprintf(stderr,
                "[sp-engine:http] /v1/chat/completions: %zu msgs, n_predict=%d, "
                "prompt_len=%zu chars\n",
                roles.size(), n_predict, prompt.size());

            const auto t0 = std::chrono::steady_clock::now();
            int rc = engine->generate(prompt, n_predict, out);
            const auto t1 = std::chrono::steady_clock::now();

            if (rc != 0) {
                res.status = 500;
                res.set_content(
                    "{\"error\":{\"message\":\"engine.generate failed\"}}",
                    "application/json");
                return;
            }

            // Strip the assistant trailing markers.
            const std::string content = strip_chatml_tail(out);

            // Build OpenAI-compatible response.
            const auto created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                t1 - t0).count();
            std::fprintf(stderr,
                "[sp-engine:http] generate done: %lld ms, %zu chars\n",
                (long long)elapsed_ms, content.size());

            std::ostringstream resp;
            resp << "{"
                 << "\"id\":\"" << gen_chatcmpl_id() << "\","
                 << "\"object\":\"chat.completion\","
                 << "\"created\":" << created << ","
                 << "\"model\":\"" << json_escape(model_name) << "\","
                 << "\"choices\":[{"
                 <<   "\"index\":0,"
                 <<   "\"message\":{"
                 <<     "\"role\":\"assistant\","
                 <<     "\"content\":\"" << json_escape(content) << "\""
                 <<   "},"
                 <<   "\"finish_reason\":\"stop\""
                 << "}],"
                 << "\"usage\":{"
                 <<   "\"prompt_tokens\":0,"
                 <<   "\"completion_tokens\":0,"
                 <<   "\"total_tokens\":0"
                 << "}"
                 << "}";

            res.set_content(resp.str(), "application/json");
        });

    // Also accept /v1/completions (legacy/text-only) — some clients use it.
    impl_->svr.Post("/v1/completions",
        [engine, &model_name](const httplib::Request& req, httplib::Response& res) {
            const std::string prompt = json_get_string(req.body, "prompt");
            const int n_predict = json_get_int(req.body, "max_tokens", 256);
            if (prompt.empty()) {
                res.status = 400;
                res.set_content(
                    "{\"error\":{\"message\":\"missing prompt\"}}",
                    "application/json");
                return;
            }
            std::string out;
            const int rc = engine->generate(prompt, n_predict, out);
            if (rc != 0) {
                res.status = 500;
                res.set_content(
                    "{\"error\":{\"message\":\"engine.generate failed\"}}",
                    "application/json");
                return;
            }
            const auto created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            std::ostringstream resp;
            resp << "{"
                 << "\"id\":\"" << gen_chatcmpl_id() << "\","
                 << "\"object\":\"text_completion\","
                 << "\"created\":" << created << ","
                 << "\"model\":\"" << json_escape(model_name) << "\","
                 << "\"choices\":[{"
                 <<   "\"text\":\"" << json_escape(out) << "\","
                 <<   "\"index\":0,"
                 <<   "\"finish_reason\":\"stop\""
                 << "}]"
                 << "}";
            res.set_content(resp.str(), "application/json");
        });

    std::fprintf(stderr,
        "[sp-engine:http] listening on %s:%d (model=%s)\n",
        host.c_str(), port, model_name.c_str());

    impl_->running.store(true);
    const bool ok = impl_->svr.listen(host.c_str(), port);
    impl_->running.store(false);

    return ok ? 0 : -1;
}

void HttpServer::stop() {
    if (impl_->running.load()) {
        impl_->svr.stop();
    }
}

} // namespace sp::engine
