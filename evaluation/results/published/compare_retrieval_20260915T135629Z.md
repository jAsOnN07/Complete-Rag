# retrieval comparison - 20260915T135629Z

- embedder: embed-v4.0
- gold: 14 questions, 0 unverified
- k: 5
- chunking: fixed 1000/150

| config | hit@1 | chunk R@5 | doc R@5 | MRR | nDCG@5 | neg gate |
|---|---|---|---|---|---|---|
| dense | 0.73 | 0.97 | 0.94 | 0.84 | 0.84 | 1.00 |
| hybrid | 0.91 | 0.97 | 0.92 | 0.95 | 0.90 | 0.00 |
| hybrid+rerank | 1.00 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 |
