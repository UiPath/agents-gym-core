"""Tool that loads a file attachment (image, PDF or text file) as LangChain message content."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, Field
from typing_extensions import override

from agents_gym_core.common_tools.file_attachments.attachment import FileAttachmentSource


class LoadAttachmentInput(BaseModel):
    """Input arguments for the LoadAttachmentTool."""

    file_attachment: FileAttachmentSource = Field(
        description="The attachment source containing a file path or URL to be loaded"
    )


class LoadAttachmentOutput(BaseModel):
    """Output from the LoadAttachmentTool."""

    content: dict[str, Any] | None = Field(
        description="The LangChain format representation of the document"
    )
    document_type: str | None = Field(description="The derived type of the document")
    file_attachment: FileAttachmentSource = Field(
        description="The attachment source containing a file path or URL to be loaded"
    )
    error: str | None = Field(default=None, description="Error message if loading failed")


class LoadAttachmentTool(BaseTool):
    """Tool for loading attachments and converting them to LangChain format.

    Supports images, PDFs, and text-based files (JSON, TXT, CSV, etc.).
    """

    name: str = "load_attachment"
    description: str = (
        "Use this tool to extract and read the text content from file attachments. "
        "Call this tool when you need to access the contents of a file provided as input "
        "(e.g., medical reports, documents, spreadsheets). "
        "Supports images, PDFs, JSON, TXT, CSV, and other text files."
    )
    args_schema: ArgsSchema | None = LoadAttachmentInput
    llm: BaseChatModel | None = Field(default=None)
    documents_dir: str = Field(
        default="",
        description="The agent's documents folder, relative to agents_gym_core/agents/ "
        "(e.g. 'my_agent/documents'). Local files are read only from it.",
    )

    @override
    def _run(
        self,
        file_attachment: FileAttachmentSource,
    ) -> list[dict[str, Any]]:
        """Load the file or URL and return it as LangChain content blocks."""
        file_attachment_results = file_attachment.to_attachment(self.documents_dir)

        langchain_content = file_attachment_results.to_langchain_content(llm=self.llm)

        # Some providers reject tool results without a non-empty text block.
        return [
            {"type": "text", "text": f"Successfully loaded attachment: {file_attachment.file}"},
            langchain_content,
        ]

    @override
    async def _arun(
        self,
        file_attachment: FileAttachmentSource,
    ) -> list[dict[str, Any]]:
        """Async wrapper around the synchronous ``_run``."""
        return self._run(file_attachment)
