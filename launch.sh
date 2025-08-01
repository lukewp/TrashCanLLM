#!/bin/bash

MVK_CONFIG_LOG_LEVEL=0 \
llama-server \
  -m ../llm-models/microsoft_Phi-4-mini-instruct-Q4_0.gguf \
  --ctx-size 65536 \
  --n-gpu-layers 32 --tensor-split 1/1 \
  --no-kv-offload \
  --batch-size 64 --ubatch-size 32 \
  --threads 24 \
  --temp 0.6 --top_p 0.95 \
  --host :: --port 8080


