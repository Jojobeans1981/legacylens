# LegacyLens AI Cost Analysis

## Summary
- Total input tokens: 11,608
- Total output tokens: 6,696
- Total cost to date: $0.0361

## Pricing Reference
- Claude Haiku 4.5 (claude-haiku-4-5-20251001): $0.80/MTok input, $4.00/MTok output
- Claude Sonnet (claude-sonnet-4-5-20250514): $3.00/MTok input, $15.00/MTok output (when available)
- Embeddings: Free (local sentence-transformers model)
- Pinecone: Free starter tier

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

## Production Projections

### Per-Query Cost (Claude Haiku 4.5)
- Average input tokens per query: ~1,451
- Average output tokens per query: ~837
- Average cost per query: $0.0045
- Estimated cost per 100 queries: $0.45
- Estimated cost per 1,000 queries: $4.50

### Per-Query Cost (Claude Sonnet, if used)
- Average cost per query: ~$0.017
- Estimated cost per 100 queries: $1.69
- Estimated cost per 1,000 queries: $16.90

### Infrastructure Costs
- Pinecone starter tier: Free (up to 100K vectors)
- Render web service: $7/mo (starter)
- Render persistent disk: $0.25/GB/mo = $0.25/mo

### Monthly Budget Estimate (100 queries/day)
- LLM API: $13.50/mo (Haiku) or $50.70/mo (Sonnet)
- Infrastructure: $7.25/mo
- **Total: ~$20.75/mo (Haiku) or ~$57.95/mo (Sonnet)**
