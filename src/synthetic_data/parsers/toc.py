"""Table of Contents (TOC) JSON parser."""

import json
from dataclasses import dataclass
from pathlib import Path

from .base import DocumentParser


@dataclass
class TOCEntry:
    """Entry in a table of contents."""

    title: str
    path: str | None = None
    children: list["TOCEntry"] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class TOCParser(DocumentParser):
    """Parser for TOC JSON files."""

    def can_parse(self, path: Path) -> bool:
        """Check if file is a TOC JSON file."""
        return path.name == "_toc.json"

    def parse(self, path: Path) -> TOCEntry:
        """Parse TOC JSON and return structured entries."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        return self._parse_entry(data)

    def _parse_entry(self, data: dict | str) -> TOCEntry:
        """Parse a single TOC entry recursively."""
        if isinstance(data, str):
            return TOCEntry(title="", path=data)
        
        title = data.get("title", "")
        path = data.get("path")
        children = []
        
        if "children" in data:
            children = [self._parse_entry(child) for child in data["children"]]
        
        return TOCEntry(title=title, path=path, children=children)

    def get_ordered_paths(self, toc: TOCEntry, base_dir: Path) -> list[Path]:
        """Get ordered list of document paths from TOC."""
        paths = []
        
        if toc.path:
            doc_path = base_dir / toc.path
            if not doc_path.suffix:
                # Try common extensions
                for ext in [".ipynb", ".mdx", ".md"]:
                    candidate = doc_path.with_suffix(ext)
                    if candidate.exists():
                        paths.append(candidate)
                        break
            elif doc_path.exists():
                paths.append(doc_path)
        
        for child in toc.children:
            paths.extend(self.get_ordered_paths(child, base_dir))
        
        return paths

