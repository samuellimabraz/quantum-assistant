"""Jupyter notebook parser."""

import re
from pathlib import Path

import nbformat

from synthetic_data.parsers.base import Document, DocumentParser, ImageReference


class JupyterParser(DocumentParser):
    """Parser for Jupyter notebook (.ipynb) files."""

    def can_parse(self, path: Path) -> bool:
        """Check if file is a Jupyter notebook."""
        return path.suffix == ".ipynb"

    def parse(self, path: Path) -> Document:
        """Parse Jupyter notebook and extract content."""
        with open(path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        title = self._extract_title(nb)
        content_parts = []
        code_blocks = []
        images = []

        for cell in nb.cells:
            if cell.cell_type == "markdown":
                md_content = cell.source
                content_parts.append(md_content)

                img_refs = self._extract_images(md_content, path)
                images.extend(img_refs)

            elif cell.cell_type == "code":
                code = cell.source.strip()
                if code:
                    code_blocks.append(code)
                    content_parts.append(f"```python\n{code}\n```")

        content = "\n\n".join(content_parts)

        metadata = {
            "kernel": nb.metadata.get("kernelspec", {}).get("name", ""),
            "language": nb.metadata.get("language_info", {}).get("name", "python"),
            "cell_count": len(nb.cells),
        }

        return Document(
            source_path=path,
            title=title,
            content=content,
            code_blocks=code_blocks,
            images=images,
            metadata=metadata,
        )

    def _extract_title(self, nb: nbformat.NotebookNode) -> str:
        """Extract title from notebook (first markdown heading)."""
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                lines = cell.source.split("\n")
                for line in lines:
                    if line.startswith("# "):
                        return line[2:].strip()
        return ""

    def _extract_images(self, markdown: str, source_path: Path) -> list[ImageReference]:
        """Extract image references from markdown content."""
        images = []

        # ![alt](path)
        pattern = r"!\[(.*?)\]\((.*?)\)"
        matches = re.finditer(pattern, markdown)

        for match in matches:
            alt_text = match.group(1)
            img_path = match.group(2)

            start = max(0, match.start() - 100)
            end = min(len(markdown), match.end() + 100)
            context = markdown[start:end].strip()

            images.append(
                ImageReference(
                    path=img_path,
                    alt_text=alt_text,
                    context=context,
                )
            )

        return images
