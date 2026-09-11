"""LlamaIndex ingestion and BM25 retrieval, with no embedding API or LLM required."""

import hashlib
import re
from pathlib import Path
from threading import RLock

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever


class KnowledgeIndex:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self._fingerprint = None
        self._retriever = None
        self._lock = RLock()
        self.node_count = 0

    def _documents(self):
        documents = []
        total = 0
        for path in sorted(self.root.glob("*.md"))[:100]:
            if path.is_symlink() or not path.is_file():
                continue
            try:
                with path.open("rb") as stream:
                    raw = stream.read(1_000_001)
                if len(raw) > 1_000_000:
                    continue
                content = raw.decode("utf-8")
            except (OSError, UnicodeError):
                continue
            total += len(raw)
            if total > 5_000_000:
                break
            if content.strip():
                documents.append(Document(
                    text=content, metadata={"source": path.name},
                    excluded_embed_metadata_keys=["source"],
                    excluded_llm_metadata_keys=["source"],
                ))
        return documents

    def _refresh(self):
        documents = self._documents()
        digest = hashlib.sha256()
        for doc in documents:
            digest.update(doc.metadata["source"].encode())
            digest.update(b"\0")
            digest.update(doc.text.encode())
            digest.update(b"\0")
        fingerprint = digest.hexdigest()
        if fingerprint == self._fingerprint:
            return
        parsed = MarkdownNodeParser(include_metadata=True).get_nodes_from_documents(documents)
        nodes = []
        for node in parsed:
            # Bound excerpts independently of tokenizers/model downloads.
            for start in range(0, len(node.text), 1000):
                chunk = node.text[start:start + 1200]
                if re.search(r"\w", chunk):
                    nodes.append(TextNode(text=chunk, metadata={"source": node.metadata["source"]},
                                          excluded_embed_metadata_keys=["source"],
                                          excluded_llm_metadata_keys=["source"]))
                if len(nodes) >= 4000:
                    break
            if len(nodes) >= 4000:
                break
        retriever = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=min(3, len(nodes)),
            skip_stemming=True, token_pattern=r"(?u)\b\w+\b", language=None,
        ) if nodes else None
        self._retriever = retriever
        self.node_count = len(nodes)
        self._fingerprint = fingerprint

    def search(self, query: str) -> dict:
        if len(query) > 2000:
            return {"error": "검색어는 2000자 이하여야 합니다.", "matches": []}
        with self._lock:
            self._refresh()
            hits = self._retriever.retrieve(query) if self._retriever and re.search(r"\w", query) else []
            return {
                "engine": "llamaindex-bm25", "indexed_nodes": self.node_count,
                "matches": [
                    {"source": hit.node.metadata["source"], "node_id": hit.node.node_id,
                     "score": float(hit.score), "content": hit.node.text}
                    for hit in hits if hit.score is not None and hit.score > 0
                ],
            }
