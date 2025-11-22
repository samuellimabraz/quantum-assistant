"""Image resolution and extraction utilities."""

import base64
import hashlib
from pathlib import Path

import fitz
import nbformat


class ImageResolver:
    """Resolve and extract images from various sources."""

    MIN_IMAGE_SIZE = 160
    MIN_IMAGE_AREA = 17000

    def __init__(self, images_output_dir: Path):
        """
        Initialize image resolver.

        Args:
            images_output_dir: Directory to save extracted images
        """
        self.images_output_dir = Path(images_output_dir)
        self.images_output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_documentation_image(self, image_path: str, source_path: Path) -> Path | None:
        """
        Resolve documentation image path to absolute path.
        Works for both MDX and Jupyter notebooks.

        Args:
            image_path: Image path (e.g., /learning/images/... or /docs/images/...)
            source_path: Path to source file

        Returns:
            Absolute path to image file, or None if not found
        """
        # Handle URLs
        if image_path.startswith(("http://", "https://")):
            return None

        # Handle absolute paths starting with /
        if image_path.startswith("/"):
            doc_root = self._find_doc_root(source_path)
            if doc_root:
                public_dir = doc_root / "public"
                if public_dir.exists():
                    candidate = public_dir / image_path.lstrip("/")
                    if candidate.exists():
                        return candidate

                    # Check for common image extensions if not found
                    for ext in [".avif", ".svg", ".png", ".jpg", ".jpeg", ".gif"]:
                        candidate_with_ext = public_dir / (image_path.lstrip("/") + ext)
                        if candidate_with_ext.exists():
                            return candidate_with_ext

        # Handle relative paths
        if not image_path.startswith("/"):
            # Try relative to source file's directory
            candidate = source_path.parent / image_path
            if candidate.exists():
                return candidate

            # Try relative to source file's parent directories (up to 3 levels)
            current = source_path.parent
            for _ in range(3):
                current = current.parent
                candidate = current / image_path
                if candidate.exists():
                    return candidate

        return None

    def extract_notebook_image(self, attachment_name: str, notebook_path: Path) -> Path | None:
        """
        Extract embedded image from notebook attachments and save to disk.

        Args:
            attachment_name: Name of attachment (e.g., "Screenshot.png")
            notebook_path: Path to notebook file

        Returns:
            Path to extracted image file, or None if extraction failed
        """
        try:
            with open(notebook_path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # Search for attachment in cells
            for cell in nb.cells:
                if cell.cell_type != "markdown":
                    continue

                attachments = cell.get("attachments", {})
                if attachment_name not in attachments:
                    continue

                # Get image data
                attachment_data = attachments[attachment_name]

                # Handle different formats
                for mime_type, data in attachment_data.items():
                    if mime_type.startswith("image/"):
                        # Generate unique filename
                        file_hash = hashlib.md5(data.encode()).hexdigest()[:8]
                        ext = mime_type.split("/")[-1]
                        if ext == "jpeg":
                            ext = "jpg"

                        output_filename = f"{notebook_path.stem}_{file_hash}.{ext}"
                        output_path = self.images_output_dir / output_filename

                        # Decode and save
                        image_bytes = base64.b64decode(data)
                        with open(output_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        return output_path

        except Exception as e:
            print(f"Warning: Failed to extract image {attachment_name} from {notebook_path}: {e}")

        return None

    def extract_data_uri_image(self, data_uri: str, source_path: Path) -> Path | None:
        """
        Extract image from data URI (base64 encoded inline image).

        Args:
            data_uri: Data URI string (data:image/png;base64,...)
            source_path: Path to source document

        Returns:
            Path to extracted image file, or None if extraction failed
        """
        try:
            if not data_uri.startswith("data:image/"):
                return None

            # Parse: data:image/png;base64,<base64_data>
            header, data = data_uri.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "")

            # Get extension
            ext = mime_type.split("/")[-1]
            if ext == "jpeg":
                ext = "jpg"

            # Decode base64
            image_bytes = base64.b64decode(data)

            # Generate unique filename using hash
            file_hash = hashlib.md5(image_bytes).hexdigest()[:8]
            output_filename = f"{source_path.stem}_inline_{file_hash}.{ext}"
            output_path = self.images_output_dir / output_filename

            # Save if doesn't exist (deduplication)
            if not output_path.exists():
                with open(output_path, "wb") as f:
                    f.write(image_bytes)

            return output_path

        except Exception as e:
            print(f"Warning: Failed to extract data URI image: {e}")
            return None

    def extract_notebook_output_image(
        self,
        image_data: str,
        image_format: str,
        notebook_path: Path,
        cell_idx: int,
        output_idx: int,
    ) -> Path | None:
        """
        Extract image from notebook cell output and save to disk.

        Args:
            image_data: Base64-encoded image data (or raw SVG string)
            image_format: Image format (png, jpeg, svg+xml)
            notebook_path: Path to notebook file
            cell_idx: Index of the cell
            output_idx: Index of the output

        Returns:
            Path to extracted image file, or None if extraction failed
        """
        try:
            # Normalize format
            ext = image_format.replace("+xml", "")
            if ext == "jpeg":
                ext = "jpg"

            # Generate unique filename
            output_filename = (
                f"{notebook_path.stem}_cell{cell_idx}_output{output_idx}.{ext}"
            )
            output_path = self.images_output_dir / output_filename

            # Handle SVG (not base64 encoded)
            if image_format == "svg+xml":
                # SVG is stored as plain text/list
                svg_content = image_data
                if isinstance(svg_content, list):
                    svg_content = "".join(svg_content)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(svg_content)
            else:
                # PNG/JPEG are base64 encoded
                # Handle both string and list of strings
                if isinstance(image_data, list):
                    image_data = "".join(image_data)

                image_bytes = base64.b64decode(image_data)

                with open(output_path, "wb") as img_file:
                    img_file.write(image_bytes)

            return output_path

        except Exception as e:
            print(
                f"Warning: Failed to extract output image from {notebook_path} "
                f"(cell {cell_idx}, output {output_idx}): {e}"
            )
            return None

    def extract_pdf_images(self, pdf_path: Path) -> list[Path]:
        """
        Extract all images from a PDF and save to disk.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of paths to extracted images
        """
        extracted_images = []

        try:
            doc = fitz.open(str(pdf_path))

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)

                        # Filter by dimensions
                        if not self._is_valid_image_size(width, height):
                            continue

                        # Generate unique filename
                        file_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                        output_filename = (
                            f"{pdf_path.stem}_p{page_num}_i{img_index}_{file_hash}.{image_ext}"
                        )
                        output_path = self.images_output_dir / output_filename

                        # Save image
                        with open(output_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        extracted_images.append(output_path)
                    except Exception as e:
                        print(
                            f"Warning: Failed to extract image {img_index} from page {page_num} of {pdf_path}: {e}"
                        )
                        continue

            doc.close()
        except Exception as e:
            print(f"Warning: Failed to extract images from PDF {pdf_path}: {e}")

        return extracted_images

    def extract_pdf_image_by_reference(self, image_ref: str, pdf_path: Path) -> Path | None:
        """
        Extract a specific image from a PDF by its reference.

        Args:
            image_ref: Reference string (e.g., "pdf:file.pdf:page0:img0")
            pdf_path: Path to PDF file

        Returns:
            Path to extracted image, or None if extraction failed
        """
        try:
            # Parse reference: pdf:filename:pageN:imgN
            parts = image_ref.split(":")
            if len(parts) != 4 or parts[0] != "pdf":
                return None

            page_num = int(parts[2].replace("page", ""))
            img_index = int(parts[3].replace("img", ""))

            doc = fitz.open(str(pdf_path))
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            if img_index < len(image_list):
                img = image_list[img_index]
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Filter by dimensions
                if not self._is_valid_image_size(width, height):
                    doc.close()
                    return None

                # Generate unique filename
                file_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                output_filename = (
                    f"{pdf_path.stem}_p{page_num}_i{img_index}_{file_hash}.{image_ext}"
                )
                output_path = self.images_output_dir / output_filename

                # Save image
                with open(output_path, "wb") as img_file:
                    img_file.write(image_bytes)

                doc.close()
                return output_path

            doc.close()
        except Exception as e:
            print(f"Warning: Failed to extract PDF image {image_ref}: {e}")

        return None

    def resolve_image_path(self, image_path: str, source_path: Path) -> Path | None:
        """
        Resolve any image path to absolute path.

        Args:
            image_path: Raw image path from document
            source_path: Path to source document

        Returns:
            Absolute path to image, or None if not found
        """
        # Handle notebook output images (already extracted)
        if image_path.startswith("notebook_output:"):
            # Format: notebook_output:filename.ipynb:cellN:outputN
            # These are already extracted during parsing, path is in resolved_path
            # This shouldn't normally be called since resolved_path is set during extraction
            return None

        # Handle PDF image references
        if image_path.startswith("pdf:"):
            return self.extract_pdf_image_by_reference(image_path, source_path)

        # Handle notebook attachments
        if image_path.startswith("attachment:"):
            attachment_name = image_path.replace("attachment:", "")
            return self.extract_notebook_image(attachment_name, source_path)

        # Handle documentation paths (both MDX and Jupyter)
        if source_path.suffix in (".mdx", ".ipynb"):
            return self.resolve_documentation_image(image_path, source_path)

        # Handle direct paths
        if Path(image_path).exists():
            return Path(image_path).resolve()

        # Try relative to source
        candidate = source_path.parent / image_path
        if candidate.exists():
            return candidate.resolve()

        return None

    def _is_valid_image_size(self, width: int, height: int) -> bool:
        """
        Check if image dimensions are valid for training.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            True if image is large enough
        """
        if width < self.MIN_IMAGE_SIZE and height < self.MIN_IMAGE_SIZE:
            return False

        area = width * height
        if area < self.MIN_IMAGE_AREA:
            return False

        return True

    def _find_doc_root(self, path: Path) -> Path | None:
        """
        Find documentation root directory containing public folder.

        Args:
            path: Path to start search from

        Returns:
            Path to doc root, or None if not found
        """
        current = path if path.is_dir() else path.parent

        # Look for common documentation root patterns
        while current != current.parent:
            public_dir = current / "public"
            if public_dir.exists() and public_dir.is_dir():
                # Verify it's a documentation root by checking for expected subdirs
                has_docs = (public_dir / "docs").exists()
                has_learning = (public_dir / "learning").exists()
                if has_docs or has_learning:
                    return current

            # Also check if we're inside a known documentation structure
            if current.name in ["qiskit-documentation", "documentation", "docs"]:
                public_dir = current / "public"
                if public_dir.exists():
                    return current

            current = current.parent

        return None
