import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field


class FileAttachmentSource(BaseModel):
    """
    Input class for creating file attachments from agent model inputs.
    Accepts file paths or URLs and converts them to proper LangchainFileAttachment objects.
    """

    file: str = Field(description="File path or URL to the data attachment content")

    def to_attachment(self, documents_dir: str = "") -> "FileAttachment":
        """Convert to a ``FileAttachment``; a local file is read only from ``documents_dir``."""
        return AttachmentConverter.convert(self, documents_dir)


class FileAttachment(BaseModel):
    """A file attachment held as a URL, base64 data or text.

    ``to_langchain_content`` turns it into a LangChain message content block.
    """

    source_type: str | None = Field(
        default=None, description="Source type: 'url', 'base64', or 'text'"
    )
    url: str | None = Field(
        default=None, description="URL for image attachments when source_type is 'url'"
    )
    data: str | None = Field(
        default=None, description="Base64 encoded data when source_type is 'base64'"
    )
    mime_type: str | None = Field(
        default=None,
        description="MIME type for base64 encoded images (e.g., 'image/jpeg', 'image/png', 'application/pdf')",
    )
    filename: str | None = Field(
        default=None,
        description="Filename for PDF attachments (required for PDFs)",
    )
    text_content: str | None = Field(
        default=None, description="Text content for text-based files (JSON, TXT, CSV, etc.)"
    )

    @classmethod
    def create_image_url(cls, url: str, mime_type: str | None = None) -> "FileAttachment":
        """Create a URL-based attachment."""
        return cls(source_type="url", url=url, mime_type=mime_type)

    @classmethod
    def create_image_base64(cls, data: str, mime_type: str) -> "FileAttachment":
        """Create a base64-encoded image attachment"""
        return cls(source_type="base64", data=data, mime_type=mime_type)

    @classmethod
    def create_pdf_base64(cls, data: str, filename: str) -> "FileAttachment":
        """Create a base64-encoded PDF attachment"""
        return cls(source_type="base64", data=data, mime_type="application/pdf", filename=filename)

    @classmethod
    def create_pdf_url(cls, url: str) -> "FileAttachment":
        """Create a URL-based PDF attachment"""
        return cls(source_type="url", url=url, mime_type="application/pdf")

    @classmethod
    def create_text_content(
        cls, text_content: str, filename: str | None = None
    ) -> "FileAttachment":
        """Create a text-based attachment (JSON, TXT, CSV, etc.)"""
        return cls(source_type="text", text_content=text_content, filename=filename)

    @staticmethod
    def _is_claude_model(llm: "BaseChatModel | None") -> bool:
        """Check if the LLM is a Claude/Anthropic model."""
        if llm is None:
            return False
        model_name = str(getattr(llm, "model_name", "")).lower()
        return "anthropic" in model_name or "claude" in model_name

    @staticmethod
    def _normalize_claude_mime_type(mime_type: str | None) -> str:
        """Return ``mime_type`` if Claude supports it, else ``image/jpeg``."""
        if not mime_type or mime_type not in ["image/png", "image/jpeg", "image/gif", "image/webp"]:
            return "image/jpeg"
        return mime_type

    def _convert_url_pdf(self) -> dict[str, Any]:
        """Convert URL-based PDF attachment."""
        return {
            "type": "file",
            "source_type": "url",
            "url": self.url,
            "mime_type": "application/pdf",
        }

    def _convert_url_image_claude(self) -> dict[str, Any]:
        """Convert URL-based image attachment for Claude models."""
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": self.url,
            },
        }

    def _convert_url_image_standard(self) -> dict[str, Any]:
        """Convert URL-based image attachment for standard models."""
        return {
            "type": "image_url",
            "image_url": {
                "url": self.url,
            },
        }

    def _convert_base64_pdf(self) -> dict[str, Any]:
        """Convert base64-encoded PDF attachment."""
        content: dict[str, Any] = {
            "type": "file",
            "source_type": "base64",
            "data": self.data,
            "mime_type": "application/pdf",
        }
        if self.filename:
            content["filename"] = self.filename
        return content

    def _convert_base64_image_claude(self) -> dict[str, Any]:
        """Convert base64-encoded image attachment for Claude models."""
        mime_type = self._normalize_claude_mime_type(self.mime_type)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": self.data,
            },
        }

    def _convert_base64_image_standard(self) -> dict[str, Any]:
        """Convert base64-encoded image attachment for standard models."""
        data_uri = f"data:{self.mime_type};base64,{self.data}"
        return {
            "type": "image_url",
            "image_url": {
                "url": data_uri,
            },
        }

    def _convert_text(self) -> dict[str, Any]:
        """Convert text-based attachment."""
        text = self.text_content or ""
        if self.filename:
            text = f"File: {self.filename}\n\n{text}"
        return {
            "type": "text",
            "text": text,
        }

    def to_langchain_content(self, llm: BaseChatModel | None = None) -> dict[str, Any]:
        """Convert to LangChain message content format."""
        is_claude = self._is_claude_model(llm)

        if self.source_type == "url":
            if self.mime_type == "application/pdf":
                return self._convert_url_pdf()
            elif is_claude and self.mime_type and self.mime_type.startswith("image/"):
                return self._convert_url_image_claude()
            else:
                return self._convert_url_image_standard()

        elif self.source_type == "base64":
            if self.mime_type == "application/pdf":
                return self._convert_base64_pdf()
            elif is_claude and self.mime_type and self.mime_type.startswith("image/"):
                return self._convert_base64_image_claude()
            else:
                return self._convert_base64_image_standard()

        elif self.source_type == "text":
            return self._convert_text()

        else:
            raise ValueError(f"Unknown file attachment type: {self.source_type}")


class AttachmentConverter:
    """Builds a ``FileAttachment`` from a ``FileAttachmentSource``."""

    @staticmethod
    def convert(source: "FileAttachmentSource", documents_dir: str = "") -> "FileAttachment":
        """Treat http(s) URLs as URL attachments and anything else as a local path."""
        parsed = urlparse(source.file)
        if parsed.scheme in ("http", "https"):
            return AttachmentConverter._create_from_url(source.file)
        else:
            return AttachmentConverter._create_from_path(source.file, documents_dir)

    @staticmethod
    def _create_from_url(file_url: str) -> "FileAttachment":
        """Create a URL attachment with the MIME type guessed from the URL."""
        mime_type, _ = mimetypes.guess_type(file_url)
        return FileAttachment.create_image_url(file_url, mime_type)

    @staticmethod
    def _create_from_path(file_path: str, documents_dir: str = "") -> "FileAttachment":
        """Read text files as text and images or PDFs as base64.

        The model chooses ``file_path``, so it is read only from ``documents_dir``: the agent's
        documents folder, relative to ``agents_gym_core/agents/``, from the tool's
        configuration. Without it no local file is read.
        """
        from agents_gym_core.common_tools.spreadsheet_tools import _resolve_data_path

        path = Path(_resolve_data_path(file_path))
        agents_dir = Path(__file__).resolve().parents[2] / "agents"
        if not documents_dir or not path.is_relative_to((agents_dir / documents_dir).resolve()):
            raise PermissionError(f"{file_path} is not in the agent's documents folder")

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Text files are passed through as text, so no MIME type is needed.
        text_extensions = {".json", ".txt", ".csv", ".md", ".xml", ".yaml", ".yml", ".log"}

        if path.suffix.lower() in text_extensions:
            try:
                with open(path, encoding="utf-8") as f:
                    text_content = f.read()
                return FileAttachment.create_text_content(text_content, path.name)
            except UnicodeDecodeError:
                raise ValueError(f"Failed to read file as text: {file_path}")

        mime_type, _ = mimetypes.guess_type(str(path))

        if mime_type and (mime_type.startswith("image/") or mime_type == "application/pdf"):
            with open(path, "rb") as f:
                file_data = f.read()
                base64_data = base64.b64encode(file_data).decode("utf-8")

            if mime_type == "application/pdf":
                return FileAttachment.create_pdf_base64(base64_data, path.name)
            else:
                return FileAttachment.create_image_base64(base64_data, mime_type)
        else:
            raise ValueError(
                f"Unsupported file type for attachment. Supported types: images, PDFs, JSON, TXT, CSV, and other text files: {path.suffix}"
            )
