# Semantic worker

The worker runs one selected Hugging Face model per process. Supply absolute paths with **--checkpoint**, **--config** and **--tokenizer**. Pass **--ipc-stdio**, **--task embedding|reranker**, **--model-id** and **--backend cpu|cuda|wgpu**. The model ID is an application identifier. The loader selects the architecture from **model_type** in **config.json**.

| Model architecture | Runtime | Embeddings | Reranking |
| --- | --- | --- | --- |
| Qwen3 dense | Burn Flex, CUDA or WGPU | Last-token pooling | Qwen3 yes/no prompt and language-model head |
| BERT, including MiniLM | Candle CPU | CLS, mean or last-token pooling | Sequence classification |
| DistilBERT | Candle CPU | CLS, mean or last-token pooling | Sequence classification |
| RoBERTa and XLM-R | Candle CPU | CLS, mean or last-token pooling | Sequence classification |

Embeddings accept **--pooling-config** with a Hugging Face Sentence Transformers **1_Pooling/config.json**, or an explicit **--pooling cls|mean|last**. Qwen3 defaults to last-token pooling. All embeddings are L2-normalized. The host can set **--query-prefix** and **--document-prefix** per model. Changing model, pooling or prefixes requires rebuilding the embedding index.

A classifier with one output logit returns its sigmoid. For several logits, supply **--positive-label** with the relevant class index. Qwen3 takes an optional **--instruction** for the yes/no prompt. Each request takes up to 32 inputs, in order. Override the default token budget of 512 with **--max-tokens** up to the model's own position limit.

## Artifacts and installation

**install-plan.json** lists tested checkpoint revisions. To select another repository, the installer can resolve the files and write a plan before starting the worker.

    python create_install_plan.py sentence-transformers/all-MiniLM-L6-v2 --task embedding --max-tokens 256
    python create_install_plan.py BAAI/bge-reranker-base --task reranker

The utility resolves the requested revision to a commit and records sizes and SHA-256 hashes. Install the listed artifacts and supply their paths as launch arguments. The worker only reads local files. It currently accepts a single Safetensors file and a tokenizer.json. Sentence Transformers models that need projection layers, custom modules or multiple pooling modes require additional adapters.

## IPC

Transport uses UTF-8 JSON lines with **version: 1** and a UUID **id**. Commands are **capabilities**, **load**, **warmup**, **status**, **infer**, **unload** and **shutdown**. Call **load** and **warmup** before sending inference.

- Embedding input includes **texts** and **purpose** (**query** or **document**). Output has **model**, **protocol** (**local-transformers-v1**), **dimensions** and **vectors**.
- Reranker input includes **query** and **candidates**. Output has **model**, **protocol** (**local-cross-encoder-v1** or **local-lm-reranker-v1**) and **scores**, in candidate order.


Send the launch model ID as **model_id** in the **load** payload. Successful lifecycle replies include **loaded: true** and **warm: true**, respectively.
