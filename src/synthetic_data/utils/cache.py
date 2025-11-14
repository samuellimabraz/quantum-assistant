"""Caching system for pipeline stages."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from synthetic_data.extractors.chunker import Chunk
from synthetic_data.parsers.base import Document, ImageReference


class PipelineCache:
    """Cache manager for pipeline stages to enable incremental processing."""

    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load cache manifest."""
        if self.manifest_file.exists():
            with open(self.manifest_file, "r") as f:
                return json.load(f)
        return {"version": "1.0", "stages": {}}

    def _save_manifest(self):
        """Save cache manifest."""
        with open(self.manifest_file, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _compute_hash(self, data: dict) -> str:
        """Compute hash for cache key."""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_stage_cache_key(self, stage: str, config: dict, source_files: list[Path]) -> str:
        """
        Generate cache key for a pipeline stage.
        
        Args:
            stage: Stage name (parse, chunk, filter, etc.)
            config: Relevant config for this stage
            source_files: Input files for this stage
            
        Returns:
            Cache key string
        """
        # Include file modification times to detect changes
        file_info = {}
        for path in source_files:
            if path.exists():
                file_info[str(path)] = {
                    "size": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                }
        
        cache_data = {
            "stage": stage,
            "config": config,
            "files": file_info,
        }
        
        return self._compute_hash(cache_data)

    def is_cached(self, stage: str, cache_key: str) -> bool:
        """Check if stage results are cached."""
        if stage not in self.manifest.get("stages", {}):
            return False
        
        stage_info = self.manifest["stages"][stage]
        if stage_info.get("key") != cache_key:
            return False
        
        # Check if cached files exist
        cache_file = self.cache_dir / f"{stage}_{cache_key[:8]}.json"
        return cache_file.exists()

    def load_documents(self, stage: str, cache_key: str) -> Optional[list[Document]]:
        """Load cached documents."""
        if not self.is_cached(stage, cache_key):
            return None
        
        cache_file = self.cache_dir / f"{stage}_{cache_key[:8]}.json"
        with open(cache_file, "r") as f:
            data = json.load(f)
        
        documents = []
        for doc_data in data["documents"]:
            # Reconstruct ImageReference objects
            images = []
            for img_data in doc_data.get("images", []):
                images.append(ImageReference(**img_data))
            
            doc = Document(
                source_path=Path(doc_data["source_path"]),
                title=doc_data.get("title", ""),
                content=doc_data.get("content", ""),
                code_blocks=doc_data.get("code_blocks", []),
                images=images,
                metadata=doc_data.get("metadata", {}),
            )
            documents.append(doc)
        
        return documents

    def save_documents(self, stage: str, cache_key: str, documents: list[Document]):
        """Save documents to cache."""
        cache_file = self.cache_dir / f"{stage}_{cache_key[:8]}.json"
        
        # Serialize documents
        data = {
            "documents": [
                {
                    "source_path": str(doc.source_path),
                    "title": doc.title,
                    "content": doc.content,
                    "code_blocks": doc.code_blocks,
                    "images": [
                        {
                            "path": img.path,
                            "alt_text": img.alt_text,
                            "caption": img.caption,
                            "context": img.context,
                            "resolved_path": img.resolved_path,
                            "transcription": img.transcription,
                        }
                        for img in doc.images
                    ],
                    "metadata": doc.metadata,
                }
                for doc in documents
            ],
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
        
        # Update manifest
        self.manifest["stages"][stage] = {
            "key": cache_key,
            "file": str(cache_file.name),
            "timestamp": data["timestamp"],
            "count": len(documents),
        }
        self._save_manifest()

    def load_chunks(self, stage: str, cache_key: str) -> Optional[list[Chunk]]:
        """Load cached chunks."""
        if not self.is_cached(stage, cache_key):
            return None
        
        cache_file = self.cache_dir / f"{stage}_{cache_key[:8]}.json"
        with open(cache_file, "r") as f:
            data = json.load(f)
        
        chunks = []
        for chunk_data in data["chunks"]:
            # Reconstruct ImageReference objects
            images = []
            for img_data in chunk_data.get("images", []):
                images.append(ImageReference(**img_data))
            
            chunk = Chunk(
                text=chunk_data["text"],
                source_path=Path(chunk_data["source_path"]),
                chunk_id=chunk_data["chunk_id"],
                code_blocks=chunk_data.get("code_blocks", []),
                images=images,
                metadata=chunk_data.get("metadata", {}),
            )
            chunks.append(chunk)
        
        return chunks

    def save_chunks(self, stage: str, cache_key: str, chunks: list[Chunk]):
        """Save chunks to cache."""
        cache_file = self.cache_dir / f"{stage}_{cache_key[:8]}.json"
        
        # Serialize chunks
        data = {
            "chunks": [
                {
                    "text": chunk.text,
                    "source_path": str(chunk.source_path),
                    "chunk_id": chunk.chunk_id,
                    "code_blocks": chunk.code_blocks,
                    "images": [
                        {
                            "path": img.path,
                            "alt_text": img.alt_text,
                            "caption": img.caption,
                            "context": img.context,
                            "resolved_path": img.resolved_path,
                            "transcription": img.transcription,
                        }
                        for img in chunk.images
                    ],
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
        
        # Update manifest
        self.manifest["stages"][stage] = {
            "key": cache_key,
            "file": str(cache_file.name),
            "timestamp": data["timestamp"],
            "count": len(chunks),
        }
        self._save_manifest()

    def clear_stage(self, stage: str):
        """Clear cache for a specific stage."""
        if stage in self.manifest.get("stages", {}):
            stage_info = self.manifest["stages"][stage]
            cache_file = self.cache_dir / stage_info["file"]
            if cache_file.exists():
                cache_file.unlink()
            del self.manifest["stages"][stage]
            self._save_manifest()

    def clear_all(self):
        """Clear all cache."""
        for file in self.cache_dir.glob("*.json"):
            if file.name != "manifest.json":
                file.unlink()
        self.manifest = {"version": "1.0", "stages": {}}
        self._save_manifest()

    def get_cache_info(self) -> dict:
        """Get information about cached stages."""
        info = {}
        for stage, stage_info in self.manifest.get("stages", {}).items():
            cache_file = self.cache_dir / stage_info["file"]
            info[stage] = {
                "timestamp": stage_info["timestamp"],
                "count": stage_info["count"],
                "size": cache_file.stat().st_size if cache_file.exists() else 0,
            }
        return info
