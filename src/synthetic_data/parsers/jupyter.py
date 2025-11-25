"""Jupyter notebook parser."""

import re
from pathlib import Path

import nbformat

from synthetic_data.parsers.base import Document, DocumentParser, ImageReference


class JupyterParser(DocumentParser):
    """Parser for Jupyter notebook (.ipynb) files."""

    def __init__(self, image_resolver=None):
        """
        Initialize parser with optional image resolver.

        Args:
            image_resolver: ImageResolver instance for extracting embedded images
        """
        self.image_resolver = image_resolver

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

        for cell_idx, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                md_content = cell.source
                content_parts.append(md_content)

                # Extract markdown images
                img_refs = self._extract_markdown_images(md_content, path)
                images.extend(img_refs)

            elif cell.cell_type == "code":
                code = cell.source.strip()
                if code:
                    code_blocks.append(code)
                    content_parts.append(f"```python\n{code}\n```")

                    # Extract output images if cell has outputs
                    if hasattr(cell, "outputs") and cell.outputs:
                        output_images = self._extract_output_images(
                            cell.outputs, path, cell_idx, code
                        )
                        images.extend(output_images)

        content = "\n\n".join(content_parts)

        metadata = {
            "kernel": nb.metadata.get("kernelspec", {}).get("name", ""),
            "language": nb.metadata.get("language_info", {}).get("name", "python"),
            "cell_count": len(nb.cells),
            "output_images": sum(1 for img in images if img.path.startswith("notebook_output:")),
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

    def _extract_markdown_images(self, markdown: str, source_path: Path) -> list[ImageReference]:
        """Extract image references from markdown content, including data URIs."""
        images = []

        # Pattern 1: Markdown format ![alt](path)
        md_pattern = r"!\[(.*?)\]\((.*?)\)"
        for match in re.finditer(md_pattern, markdown):
            alt_text = match.group(1)
            img_path = match.group(2).strip()

            # ![alt](url "title")
            if ' "' in img_path:
                img_path = img_path.split(' "')[0]
            elif " '" in img_path:
                img_path = img_path.split(" '")[0]

            start = max(0, match.start() - 100)
            end = min(len(markdown), match.end() + 100)
            context = markdown[start:end].strip()

            # Handle data URI image
            if img_path.startswith("data:image/") and self.image_resolver:
                extracted_path = self.image_resolver.extract_data_uri_image(img_path, source_path)
                images.append(
                    ImageReference(
                        path=img_path[:50] + "..." if len(img_path) > 50 else img_path,
                        alt_text=alt_text or "Inline image",
                        context=context,
                        resolved_path=str(extracted_path) if extracted_path else None,
                    )
                )
            else:
                images.append(
                    ImageReference(
                        path=img_path,
                        alt_text=alt_text,
                        context=context,
                    )
                )

        # Pattern 2: HTML format <img src="path" alt="text" ...>
        html_pattern = r'<img[^>]+src=["\'](.*?)["\'](?:[^>]+alt=["\'](.*?)["\'])?[^>]*>'
        for match in re.finditer(html_pattern, markdown):
            img_path = match.group(1).strip()
            alt_text = match.group(2) if match.group(2) else ""

            start = max(0, match.start() - 100)
            end = min(len(markdown), match.end() + 100)
            context = markdown[start:end].strip()

            # Handle data URI in HTML img tags
            if img_path.startswith("data:image/") and self.image_resolver:
                extracted_path = self.image_resolver.extract_data_uri_image(img_path, source_path)
                images.append(
                    ImageReference(
                        path=img_path[:50] + "..." if len(img_path) > 50 else img_path,
                        alt_text=alt_text or "HTML inline image",
                        context=context,
                        resolved_path=str(extracted_path) if extracted_path else None,
                    )
                )
            else:
                images.append(
                    ImageReference(
                        path=img_path,
                        alt_text=alt_text,
                        context=context,
                    )
                )

        return images

    def _extract_output_images(
        self,
        outputs: list,
        notebook_path: Path,
        cell_idx: int,
        cell_code: str,
    ) -> list[ImageReference]:
        """
        Extract images from cell outputs.

        Args:
            outputs: List of cell outputs
            notebook_path: Path to notebook file
            cell_idx: Index of the cell
            cell_code: Source code of the cell

        Returns:
            List of ImageReference objects for output images
        """
        images = []

        for output_idx, output in enumerate(outputs):
            # Check for display_data or execute_result outputs
            if output.get("output_type") not in ("display_data", "execute_result"):
                continue

            data = output.get("data", {})

            # Look for image MIME types
            image_data = None
            image_format = None

            for mime_type in ["image/png", "image/jpeg", "image/svg+xml"]:
                if mime_type in data:
                    image_data = data[mime_type]
                    image_format = mime_type.split("/")[-1]
                    break

            if not image_data or not self.image_resolver:
                continue

            # Extract and save the image
            extracted_path = self.image_resolver.extract_notebook_output_image(
                image_data=image_data,
                image_format=image_format,
                notebook_path=notebook_path,
                cell_idx=cell_idx,
                output_idx=output_idx,
            )

            if extracted_path:
                # Create context from surrounding code
                context = self._create_output_context(cell_code, output)

                # Generate descriptive alt text based on code
                alt_text = self._generate_output_alt_text(cell_code, output_idx)

                # Create reference with special path format
                ref_path = f"notebook_output:{notebook_path.name}:cell{cell_idx}:output{output_idx}"

                images.append(
                    ImageReference(
                        path=ref_path,
                        alt_text=alt_text,
                        context=context,
                        resolved_path=str(extracted_path),
                    )
                )

        return images

    def _create_output_context(self, cell_code: str, output: dict) -> str:
        """
        Create context for output image from cell code and output.

        Args:
            cell_code: Source code that generated the output
            output: Output dictionary

        Returns:
            Context string describing the image
        """
        # Use the code that generated the output as primary context
        code_preview = cell_code[:200] if len(cell_code) > 200 else cell_code

        text_data = output.get("data", {}).get("text/plain", "")
        if text_data:
            text_preview = (
                text_data[:100]
                if isinstance(text_data, str) and len(text_data) > 100
                else text_data
            )
            return f"Code: {code_preview}\nOutput: {text_preview}"

        return f"Code: {code_preview}"

    def _generate_output_alt_text(self, cell_code: str, output_idx: int) -> str:
        """
        Generate descriptive alt text for output image based on code.

        Args:
            cell_code: Source code that generated the output
            output_idx: Index of the output

        Returns:
            Descriptive alt text
        """
        # Try to infer what the image represents from common patterns
        code_lower = cell_code.lower()

        if "circuit" in code_lower and "draw" in code_lower:
            return "Quantum circuit diagram"
        elif "plot" in code_lower or "draw" in code_lower:
            return f"Visualization output {output_idx + 1}"
        elif "histogram" in code_lower:
            return "Histogram plot"
        elif "statevector" in code_lower and "draw" in code_lower:
            return "State vector visualization"
        elif "bloch" in code_lower:
            return "Bloch sphere representation"
        else:
            return f"Cell output visualization {output_idx + 1}"
