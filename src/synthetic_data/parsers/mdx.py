"""MDX file parser."""

import re
from pathlib import Path

from synthetic_data.parsers.base import Document, DocumentParser, ImageReference


class MDXParser(DocumentParser):
    """Parser for MDX (Markdown with JSX) files."""

    def can_parse(self, path: Path) -> bool:
        """Check if file is an MDX file."""
        return path.suffix == ".mdx"

    def parse(self, path: Path) -> Document:
        """Parse MDX file and extract content."""
        with open(path, encoding="utf-8") as f:
            content = f.read()

        title = self._extract_title(content)
        cleaned_content = self._clean_jsx(content)
        code_blocks = self._extract_code_blocks(cleaned_content)
        images = self._extract_images(content, path)
        metadata = self._extract_frontmatter(content)

        return Document(
            source_path=path,
            title=title,
            content=cleaned_content,
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
        """Extract code blocks from markdown content."""
        code_blocks = []

        pattern = r"```(?:\w+)?\n(.*?)```"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            code = match.group(1).strip()
            if code:
                code_blocks.append(code)

        return code_blocks

    def _extract_images(self, content: str, source_path: Path) -> list[ImageReference]:
        """Extract image references from content."""
        images = []

        # Markdown images
        md_pattern = r"!\[(.*?)\]\((.*?)\)"
        for match in re.finditer(md_pattern, content):
            img_path = match.group(2).strip()
            
            # Handle optional title
            if ' "' in img_path:
                img_path = img_path.split(' "')[0]
            elif " '" in img_path:
                img_path = img_path.split(" '")[0]

            images.append(
                ImageReference(
                    path=img_path,
                    alt_text=match.group(1),
                )
            )

        # JSX Image components
        jsx_pattern = r'<Image\s+src="([^"]+)"(?:\s+alt="([^"]+)")?'
        for match in re.finditer(jsx_pattern, content):
            images.append(
                ImageReference(
                    path=match.group(1),
                    alt_text=match.group(2) or "",
                )
            )

        return images
