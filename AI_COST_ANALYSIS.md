# LegacyLens AI Cost Analysis

## Summary
- Total input tokens: 26,843
- Total output tokens: 10,957
- Total cost to date: $0.0685
- Embedding cost: $0.00 (Pinecone Inference API included with free tier)

## Pricing Reference
- Claude Haiku 4.5 (claude-haiku-4-5-20251001): $0.80/MTok input, $4.00/MTok output
- Embeddings: Free (Pinecone Inference API, multilingual-e5-large)
- Pinecone: Free starter tier (330 vectors, 1024 dimensions)

## Per-Call Log
| Date | Operation | Input Tokens | Output Tokens | Cost | Running Total |
|------|-----------|-------------|---------------|------|---------------|
| 2026-03-02 | Test Q1: main entry point | 148 | 160 | $0.0008 | $0.0008 |
| 2026-03-02 | Test Q2: matrix modifications | 2,361 | 333 | $0.0032 | $0.0040 |
| 2026-03-02 | Test Q3: explain DGEMM | 2,233 | 630 | $0.0043 | $0.0083 |
| 2026-03-02 | Test Q4: file I/O operations | 1,829 | 403 | $0.0031 | $0.0114 |
| 2026-03-02 | Test Q5: DGEMV dependencies | 533 | 215 | $0.0013 | $0.0127 |
| 2026-03-02 | Test Q6: error handling patterns | 1,667 | 1,176 | $0.0060 | $0.0187 |
| 2026-03-02 | Test Q7: docgen SGEMM | 579 | 1,985 | $0.0084 | $0.0271 |
| 2026-03-02 | Test Q8: translate DTRSM | 2,258 | 1,794 | $0.0090 | $0.0361 |
| 2026-03-03 | Prod Q0: what is blas | 1,935 | 296 | $0.0031 | $0.0392 |
| 2026-03-03 | Prod Q1: main entry point | 1,920 | 348 | $0.0029 | $0.0422 |
| 2026-03-03 | Prod Q2: matrix modifications | 2,225 | 413 | $0.0034 | $0.0456 |
| 2026-03-03 | Prod Q3: explain DGEMM | 2,294 | 570 | $0.0041 | $0.0497 |
| 2026-03-03 | Prod Q4: file I/O ops | 2,074 | 380 | $0.0032 | $0.0529 |
| 2026-03-03 | Prod Q5: DGEMV dependencies | 2,242 | 257 | $0.0028 | $0.0557 |
| 2026-03-03 | Prod Q6: error handling | 2,140 | 602 | $0.0041 | $0.0599 |
| 2026-03-03 | Prod Q8: translate DTRSM | 2,340 | 1,691 | $0.0086 | $0.0685 |

## Production Projections

### Per-Query Cost (Claude Haiku 4.5)
- Average input tokens per query: ~2,146
- Average output tokens per query: ~507
- Average cost per query: $0.0038
- Estimated cost per 100 queries: $0.38
- Estimated cost per 1,000 queries: $3.80

### Infrastructure Costs
- Pinecone starter tier: Free (up to 100K vectors, includes inference API)
- Render web service: $7/mo (starter) or free tier
- Render persistent disk: $0.25/GB/mo = $0.25/mo

### Monthly Budget Estimate (100 queries/day)
- LLM API: $11.40/mo (Haiku)
- Infrastructure: $7.25/mo
- **Total: ~$18.65/mo (Haiku)**
