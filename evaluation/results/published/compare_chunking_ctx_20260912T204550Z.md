# chunking_ctx comparison - 20260912T204550Z

- embedder: BAAI/bge-small-en-v1.5
- gold: 14 questions, 0 unverified
- k: 5
- retrieval: hybrid + cross_encoder
- chunk_size/overlap: 1000/150

| config | hit@1 | chunk R@5 | doc R@5 | MRR | nDCG@5 | neg gate | chunks |
|---|---|---|---|---|---|---|---|
| fixed | 1.00 | 0.98 | 0.98 | 1.00 | 0.99 | 1.00 | 553 |
| fixed+ctx | 0.91 | 0.98 | 0.92 | 0.93 | 0.90 | 1.00 | 553 |
| recursive | 0.82 | 0.98 | 0.98 | 0.91 | 0.92 | 1.00 | 545 |
| recursive+ctx | 0.91 | 0.98 | 0.90 | 0.93 | 0.88 | 1.00 | 545 |
| semantic | 0.64 | 0.80 | 0.98 | 0.73 | 0.76 | 1.00 | 813 |
| semantic+ctx (not indexed) | - | - | - | - | - | - | - |
