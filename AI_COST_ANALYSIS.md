# GRIMOIRE AI Cost Analysis

## Development & Testing Costs

### Actual Spend During Development
| Cost Line | Amount |
|-----------|--------|
| Embedding API | $0.00 (Pinecone Inference API, bundled free) |
| LLM API (Claude Haiku 4.5) | $0.43 (197 queries during dev/test) |
| Vector DB (Pinecone) | $0.00 (serverless free tier) |
| Hosting (Render) | $7.00/mo (starter web service) |
| Persistent disk | $0.25/mo (1 GB) |
| **Total dev spend** | **$0.43** (LLM only; infra billed monthly) |

### Token Usage
- Total input tokens: 178,660
- Total output tokens: 46,693
- Average input tokens per query: ~907
- Average output tokens per query: ~237
- Average cost per query: $0.0022

### Pricing Reference
- Claude Haiku 4.5: $1.00/MTok input, $5.00/MTok output
- Embeddings: Free (Pinecone Inference API, multilingual-e5-large, 1024-dim)
- Pinecone: Free serverless tier (sufficient for 5,074 vectors)
- Render: $7/mo starter tier with persistent disk

## Production Cost Projections

### Assumptions
- Average query: ~900 input tokens, ~240 output tokens
- Cost per query: ~$0.002 (Haiku 4.5)
- Queries per user per day: 5
- Embedding cost for new code: $0 (Pinecone Inference bundled)
- Vector DB storage scales linearly; current 5,074 vectors well within free tier

### Monthly Cost at Scale

| Scale | Users | Queries/mo | LLM Cost | Infra Cost | Total |
|-------|-------|-----------|----------|------------|-------|
| Dev/Demo | 1-5 | ~500 | $1.00 | $7.25 | **$8.25/mo** |
| 100 Users | 100 | 15,000 | $30.00 | $7.25 | **$37.25/mo** |
| 1,000 Users | 1,000 | 150,000 | $300.00 | $25.00 | **$325.00/mo** |
| 10,000 Users | 10,000 | 1,500,000 | $3,000.00 | $100.00 | **$3,100.00/mo** |
| 100,000 Users | 100,000 | 15,000,000 | $30,000.00 | $500.00 | **$30,500.00/mo** |

### Infrastructure Notes at Scale
- **100 users**: Pinecone free tier sufficient; Render starter ($7/mo)
- **1,000 users**: Pinecone standard tier (~$10/mo); Render professional ($25/mo)
- **10,000 users**: Pinecone standard (~$70/mo); dedicated hosting (~$100/mo); consider caching aggressively
- **100,000 users**: Pinecone enterprise; multi-region hosting; query cache critical (repeat queries at ~0 cost); consider switching to local embedding model to reduce latency

### Cost Optimization Strategies
- **Query caching**: SHA-256 keyed cache (already implemented) — repeat queries cost $0
- **Shorter outputs**: Max tokens reduced from 1024 to 512 — cuts output cost ~50%
- **Compact context**: 400 chars/chunk cap — reduces input tokens per query
- **Model selection**: Haiku 4.5 is 3x cheaper than Sonnet for comparable quality on code tasks
- **Hybrid search**: BM25 keyword search is free (local) — reduces reliance on vector DB calls
