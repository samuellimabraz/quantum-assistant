"""MDX file parser."""

import hashlib
import re
from pathlib import Path

from synthetic_data.parsers.base import Document, DocumentParser, ImageReference


class MDXParser(DocumentParser):
    """Parser for MDX (Markdown with JSX) files."""

    def can_parse(self, path: Path) -> bool:
        """Check if file is an MDX file."""
        return path.suffix == ".mdx"

    def parse(self, path: Path) -> Document:
        """Parse MDX file and extract content.

        Images are extracted and replaced with [IMAGE:id] markers.
        """
        with open(path, encoding="utf-8") as f:
            content = f.read()

        title = self._extract_title(content)
        cleaned_content = self._clean_jsx(content)
        code_blocks = self._extract_code_blocks(cleaned_content)

        # Extract images and replace with markers
        content_with_markers, images = self._extract_and_replace_images(cleaned_content, path)

        metadata = self._extract_frontmatter(content)

        return Document(
            source_path=path,
            title=title,
            content=content_with_markers,
            code_blocks=code_blocks,
            images=images,
            metadata=metadata,
        )

    def _extract_title(self, content: str) -> str:
        """Extract title from first heading."""
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
        return ""

    def _extract_frontmatter(self, content: str) -> dict:
        """Extract YAML frontmatter if present."""
        if not content.startswith("---"):
            return {}

        try:
            import yaml

            end_idx = content.find("---", 3)
            if end_idx == -1:
                return {}

            frontmatter = content[3:end_idx].strip()
            return yaml.safe_load(frontmatter) or {}
        except Exception:
            return {}

    def _clean_jsx(self, content: str) -> str:
        """Remove JSX components and clean up content."""
        content = re.sub(r"^import\s+.*?$", "", content, flags=re.MULTILINE)

        # Remove frontmatter
        if content.startswith("---"):
            end_idx = content.find("---", 3)
            if end_idx != -1:
                content = content[end_idx + 3 :]

        # Convert common JSX components to markdown
        # <Image src="..." alt="..." /> -> ![alt](src)
        content = re.sub(
            r'<Image\s+src="([^"]+)"\s+alt="([^"]+)"\s*/?>',
            r"![\2](\1)",
            content,
        )

        # Remove other JSX tags but keep content
        content = re.sub(r"<[A-Z][^>]*>", "", content)
        content = re.sub(r"</[A-Z][^>]*>", "", content)

        return content.strip()

    def _extract_code_blocks(self, content: str) -> list[str]:
        """Extract code blocks from markdown content.

        Only extracts blocks with explicit language identifiers (python, javascript, etc).
        Generic ``` blocks without language are considered non-code content.
        """
        code_blocks = []

        # Only match code blocks with explicit language identifier
        pattern = r"```(\w+)\n(.*?)```"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            language = match.group(1).lower()
            code = match.group(2).strip()

            # Accept common programming languages
            if code and language in (
                "python",
                "javascript",
                "typescript",
                "java",
                "rust",
                "go",
                "cpp",
                "c",
                "sql",
                "bash",
                "sh",
                "r",
            ):
                code_blocks.append(code)

        return code_blocks

    def _generate_image_id(self, img_path: str, source_path: Path) -> str:
        """Generate unique image ID."""
        unique_str = f"{source_path.name}:{img_path}"
        hash_val = hashlib.md5(unique_str.encode()).hexdigest()[:12]
        return f"img_{hash_val}"

    def _extract_and_replace_images(
        self, content: str, source_path: Path
    ) -> tuple[str, list[ImageReference]]:
        """Extract images from content and replace with [IMAGE:id] markers.

        Returns:
            Tuple of (content_with_markers, list_of_image_references)
        """
        images = []

        # Pattern 1: Markdown images ![alt](path)
        md_pattern = r"!\[(.*?)\]\(([^)]+)\)"
        for match in re.finditer(md_pattern, content):
            alt_text = match.group(1)
            img_path = match.group(2).strip()

            # Handle optional title
            if ' "' in img_path:
                img_path = img_path.split(' "')[0]
            elif " '" in img_path:
                img_path = img_path.split(" '")[0]

            # Skip data URIs (shouldn't happen in MDX but safety check)
            if img_path.startswith("data:image/"):
                img_id = self._generate_image_id(img_path[:100], source_path)
                images.append(
                    ImageReference(
                        path=f"data_uri:{img_id}",
                        alt_text=alt_text or "Inline image",
                        image_id=img_id,
                    )
                )
            else:
                img_id = self._generate_image_id(img_path, source_path)
                images.append(
                    ImageReference(
                        path=img_path,
                        alt_text=alt_text,
                        image_id=img_id,
                    )
                )

            # Replace with marker
            marker = f"[IMAGE:{img_id}]"
            content = content.replace(match.group(0), marker, 1)

        # Pattern 2: JSX Image components <Image src="..." alt="..." />
        jsx_pattern = r'<Image\s+src="([^"]+)"(?:\s+alt="([^"]+)")?[^>]*/>'
        for match in re.finditer(jsx_pattern, content):
            full_match = match.group(0)
            img_path = match.group(1)
            alt_text = match.group(2) or ""

            img_id = self._generate_image_id(img_path, source_path)
            images.append(
                ImageReference(
                    path=img_path,
                    alt_text=alt_text,
                    image_id=img_id,
                )
            )

            marker = f"[IMAGE:{img_id}]"
            content = content.replace(full_match, marker, 1)

        return content, images
