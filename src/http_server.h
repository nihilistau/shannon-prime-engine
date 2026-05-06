// Shannon-Prime Engine — minimal OpenAI-compatible HTTP server.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Wraps Engine::generate behind /v1/chat/completions + /v1/models so
// sp-engine becomes drop-in compatible with the FastAPI proxy at
// projectx (lms_custom_proxy.py) currently routing the on-phone tier
// to llama.cpp's server at phone:8082.
//
// Phase 3.7 minimal viable: non-streaming chat, ChatML template for
// Qwen2/3 (hardcoded — follow-up reads from GGUF chat_template).
// Streaming SSE / sampling chain / tool_calls in later iterations.

#pragma once

#include "engine.h"

#include <atomic>
#include <memory>
#include <string>

namespace sp::engine {

class HttpServer {
public:
    HttpServer();
    ~HttpServer();

    // Bind to the given engine (must outlive the server). Engine::load
    // should already have been called. web_root is the directory to serve
    // static files from (optional).
    void bind(Engine* engine, const std::string& model_name, const std::string& web_root = "");

    // Start listening on host:port. Blocks until stop() is called from
    // another thread or the process exits. Returns 0 on clean shutdown,
    // non-zero on bind failure.
    int listen_and_serve(const std::string& host, int port);

    // Stop the server (call from another thread / signal handler).
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sp::engine
