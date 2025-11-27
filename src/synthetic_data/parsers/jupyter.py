"""Jupyter notebook parser."""

import hashlib
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
        """Parse Jupyter notebook and extract content.

        Images (markdown, HTML, data URIs) are extracted as ImageReference objects
        and replaced with [IMAGE:id] markers in the content.
        """
        with open(path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        title = self._extract_title(nb)
        content_parts = []
        code_blocks = []
        images = []

        for cell_idx, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                md_content = cell.source

                # Extract images and replace with markers
                md_content, img_refs = self._extract_and_replace_images(md_content, path)
                images.extend(img_refs)
                content_parts.append(md_content)

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

                        # Insert image markers after the code block
                        if output_images:
                            image_markers = [
                                f"[IMAGE:{img.image_id}]" for img in output_images if img.image_id
                            ]
                            if image_markers:
                                # Add markers as separate content part (will be joined with \n\n)
                                content_parts.append("\n".join(image_markers))

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

    def _generate_image_id(self, img_path: str, source_path: Path) -> str:
        """Generate unique image ID."""
        unique_str = f"{source_path.name}:{img_path}"
        hash_val = hashlib.md5(unique_str.encode()).hexdigest()[:12]
        return f"img_{hash_val}"

    def _extract_and_replace_images(
        self, markdown: str, source_path: Path
    ) -> tuple[str, list[ImageReference]]:
        """Extract images from markdown and replace with [IMAGE:id] markers.

        Handles:
        - Markdown images: ![alt](path)
        - HTML images: <img src="path" ...>
        - Data URIs (both markdown and HTML)

        Returns:
            Tuple of (content_with_markers, list_of_image_references)
        """
        images = []
        content = markdown

        # Pattern 1: Markdown format ![alt](path) - including data URIs
        md_pattern = r"!\[(.*?)\]\((data:image/[^)]+|[^)]+)\)"
        for match in re.finditer(md_pattern, markdown):
            alt_text = match.group(1)
            img_path = match.group(2).strip()

            # Handle optional title in path
            if not img_path.startswith("data:image/"):
                if ' "' in img_path:
                    img_path = img_path.split(' "')[0]
                elif " '" in img_path:
                    img_path = img_path.split(" '")[0]

            # Generate unique ID
            if img_path.startswith("data:image/"):
                img_id = self._generate_image_id(img_path[:100], source_path)
            else:
                img_id = self._generate_image_id(img_path, source_path)

            # Get context around the image
            start = max(0, match.start() - 100)
            end = min(len(markdown), match.end() + 100)
            context = markdown[start:end].strip()

            # Create image reference
            if img_path.startswith("data:image/"):
                resolved = None
                if self.image_resolver:
                    resolved = self.image_resolver.extract_data_uri_image(img_path, source_path)
                images.append(
                    ImageReference(
                        path=f"data_uri:{img_id}",
                        alt_text=alt_text or "Inline image",
                        context=context,
                        resolved_path=str(resolved) if resolved else None,
                        image_id=img_id,
                    )
                )
            else:
                images.append(
                    ImageReference(
                        path=img_path,
                        alt_text=alt_text,
                        context=context,
                        image_id=img_id,
                    )
                )

            # Replace the full match with marker
            marker = f"[IMAGE:{img_id}]"
            content = content.replace(match.group(0), marker, 1)

        # Pattern 2: HTML format <img src="..." ...> - including data URIs
        html_pattern = r"<img[^>]+src=[\"']([^\"']+)[\"'][^>]*/?>"
        for match in re.finditer(html_pattern, content):
            full_match = match.group(0)
            img_path = match.group(1).strip()

            # Skip if already processed (marker already in content)
            if "[IMAGE:" in full_match:
                continue

            # Extract alt text if present
            alt_match = re.search(r'alt=["\']([^"\']*)["\']', full_match)
            alt_text = alt_match.group(1) if alt_match else ""

            # Generate unique ID
            if img_path.startswith("data:image/"):
                img_id = self._generate_image_id(img_path[:100], source_path)
            else:
                img_id = self._generate_image_id(img_path, source_path)

            # Get context
            start = max(0, match.start() - 100)
            end = min(len(content), match.end() + 100)
            context_text = content[start:end].strip()

            # Create image reference
            if img_path.startswith("data:image/") and self.image_resolver:
                extracted_path = self.image_resolver.extract_data_uri_image(img_path, source_path)
                images.append(
                    ImageReference(
                        path=f"data_uri:{img_id}",
                        alt_text=alt_text or "HTML inline image",
                        context=context_text,
                        resolved_path=str(extracted_path) if extracted_path else None,
                        image_id=img_id,
                    )
                )
            else:
                images.append(
                    ImageReference(
                        path=img_path,
                        alt_text=alt_text,
                        context=context_text,
                        image_id=img_id,
                    )
                )

            # Replace with marker
            marker = f"[IMAGE:{img_id}]"
            content = content.replace(full_match, marker, 1)

        return content, images

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
                img_id = self._generate_image_id(ref_path, notebook_path)

                images.append(
                    ImageReference(
                        path=ref_path,
                        alt_text=alt_text,
                        context=context,
                        resolved_path=str(extracted_path),
                        image_id=img_id,
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
            Context string describing the image (limited to prevent API errors)
        """
        # Limit code preview to prevent huge contexts
        code_preview = cell_code[:300] if len(cell_code) > 300 else cell_code

        text_data = output.get("data", {}).get("text/plain", "")
        if text_data:
            # Limit output preview
            if isinstance(text_data, str):
                text_preview = text_data[:200] if len(text_data) > 200 else text_data
            else:
                text_preview = str(text_data)[:200]
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
