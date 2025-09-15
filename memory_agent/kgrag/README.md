# KGrag

![kgrag](./kgrag.jpg)

LLMs show strong reasoning abilities but struggle to connect information in ways that feel intuitive; traditional RAG solutions often fail to synthesize complex information correctly.

`memory-agent` combines symbolic knowledge stored in [Neo4j](https://neo4j.com/) with vector semantic search in [Qdrant](https://qdrant.tech/). Neo4j holds entities, relationships, ontological constraints and enables graph reasoning (paths, neighborhoods, patterns), while Qdrant indexes embeddings of text chunks and node descriptions for high‑similarity semantic retrieval.

- **Typical flow**: entity linking → retrieval of relevant subgraphs (Neo4j) → vector search over chunks (Qdrant) → result fusion and re‑ranking → controlled generation with citations and, when useful, executable plans (Cypher).
- **Cross‑indexing**: maintain mappings between chunk IDs and node/edge IDs to hybridize graph and vector results.
- **Benefits**: explainability and logical constraints from the graph, robustness on unstructured text via embeddings, and more accurate answers by fusing symbolic and semantic signals.
- **Use cases**: path‑based explanations (Path‑RAG), serialization of triples for contexts (Subgraph‑RAG), and applications that require formal verification via graph queries.

## 1 - Data Ingestion

In the ingestion pipeline we extract and normalize documents (PDF/HTML/text) and create context‑windowed chunks to preserve local coherence. We then perform named‑entity recognition and relation extraction to detect entities (people, organizations, places, products, concepts) and relations (for example, works_for, located_in, depends_on). Detected mentions are aligned to unique URIs through entity linking and aliases are deduplicated; from there the system produces (subject, predicate, object) triples according to an ontology schema (classes, properties, constraints). Updates are applied via idempotent upserts that merge changes atomically and record versioning and provenance metadata (source, timestamp, author). The pipeline computes hybrid embeddings for text chunks and optionally for graph nodes/edges (node2vec/GNN or text embeddings of labels/descriptions), cross‑indexes chunk IDs with graph node/edge IDs to enable hybrid graph+vector retrieval, and provides observability through structured logs and metrics (entity count, density, local diameter) as well as extraction quality and drift monitoring.

### Example

- [demo_config.py](../../demo_config.py)
- [demo_kgrag_ollama.py](../../demo_kgrag_ollama.py)

```python
import asyncio
from demo_kgrag_ollama import kgrag_ollama


async def ingestion(path: str):
    async for d in kgrag_ollama.process_documents(
        path=path,
        force=True
    ):
        if d == "ERROR":
            return f"Error processing document {path}."
    return f"Document {path} ingested successfully."


async def main():
    path = "/Users/giuseppezileni/arxiv/2508.20435v1.pdf"
    result = await ingestion(path)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

```

## 2 - Query

The goal is to answer complex questions by combining graph-based reasoning with semantic similarity. At query time the pipeline first determines intent, links user mentions to existing entities, and searches relevant subgraphs (for example k‑hop neighborhoods, pattern or temporal paths); it then retrieves semantically relevant text chunks using vector search and fuses graph and vector results (e.g., reciprocal rank fusion or cross‑encoder re‑ranking). A hierarchical context is assembled—graph facts and attributes first, then the most relevant text extracts with provenance—and the generator produces controlled outputs constrained by citations, formats, step‑by‑step reasoning and, when useful, executable graph query plans (Cypher/SPARQL) for verification. Useful patterns include Path‑RAG for explanatory paths, Subgraph‑RAG for serialized triple contexts, constraint‑guided RAG that enforces ontology rules, and a feedback loop where validated answers enrich the graph.

### Example

- [demo_config.py](../../demo_config.py)
- [demo_kgrag_ollama.py](../../demo_kgrag_ollama.py)

```python
import asyncio
from demo_kgrag_ollama import kgrag_ollama


async def main():
    prompt = (
        "How Big Data Dilutes Cognitive Resources, Interferes with Rational "
        "Decision-making and Affects Wealth Distribution ?"
    )
    response = await kgrag_ollama.query(prompt)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```
