# Start the llama.cpp LLM server for NPChat.
# Recipe B from mylaptop.md: single-user RAG (-ngl 99, ctx 8192, 2 slots).
# Embeddings run in-process — no second server needed.

& "C:\Tools\llama.cpp\llama-server.exe" `
  -m "C:\Tools\Qwen2.5-7B-Instruct-Q4_K_M.gguf" `
  --alias "qwen2.5-7b-instruct" `
  --port 8080 `
  -ngl 99 `
  --ctx-size 8192 `
  --parallel 1
