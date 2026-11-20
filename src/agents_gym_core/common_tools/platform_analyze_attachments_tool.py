"""Attachment analysis tool.

Answers queries about images, PDFs and text attachments with a dedicated model call.
Supports text responses and configured structured outputs, with attachment content supplied
in user messages.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, Field
from typing_extensions import override

from agents_gym_core.common_tools.file_attachments import FileAttachmentSource
from agents_gym_core.models.chat import get_chat_model
from agents_gym_core.utils.gym_settings_core import gym_settings_core


class PlatformAnalyzeAttachmentInput(BaseModel):
    """Generic input schema for the Platform Analyze Attachments tool."""

    query: str = Field(description="The query or task to send to the LLM")
    file_attachment: FileAttachmentSource | None = Field(
        default=None,
        description="File attachment (image, document, etc.) to include with the query; required",
    )


class PlatformAnalyzeAttachmentOutput(BaseModel):
    """Generic output schema for the Platform Analyze Attachments tool."""

    response: str = Field(description="The LLM's response to the query")


class PlatformAnalyzeAttachmentsTool(BaseTool):
    """Send a query and a required file attachment to an LLM with ``system_instructions``.

    Base64 images, PDFs and text files are supported. With ``structured_output_schema``
    the LLM's structured output is returned instead of a ``PlatformAnalyzeAttachmentOutput``.
    """

    name: str = "analyze_attachments"
    description: str = (
        "Use this tool to extract, analyze, and answer questions about file attachments. "
        "Call this tool when you need to read and understand the contents of documents provided as input "
        "(e.g., medical reports, invoices, images, spreadsheets). "
        "Supports images, PDFs, JSON, TXT, CSV, and other text files."
    )
    system_instructions: str = Field(
        default="Process the provided files to complete the given task. Analyze the files contents thoroughly to deliver an accurate response based on the extracted information."
    )

    llm: BaseChatModel | None = Field(default=None)
    model_name: str | None = Field(default=None)
    structured_output_schema: type[BaseModel] | None = Field(default=None)
    documents_dir: str = Field(
        default="",
        description="The agent's documents folder, relative to agents_gym_core/agents/ "
        "(e.g. 'my_agent/documents'). Local files are read only from it.",
    )
    args_schema: ArgsSchema | None = PlatformAnalyzeAttachmentInput

    def __init__(
        self,
        name: str = "analyze_attachments",
        description: str | None = None,
        system_instructions: str | None = None,
        llm: BaseChatModel | None = None,
        model_name: str | None = None,
        structured_output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ):
        if description is not None:
            kwargs["description"] = description
        if system_instructions is not None:
            kwargs["system_instructions"] = system_instructions
        if "args_schema" not in kwargs:
            kwargs["args_schema"] = PlatformAnalyzeAttachmentInput

        super().__init__(
            name=name,
            llm=llm,
            model_name=model_name,
            structured_output_schema=structured_output_schema,
            **kwargs,
        )

    def _get_llm(self) -> BaseChatModel:
        """Return ``llm``, or a model built from ``model_name``."""
        if self.llm is not None:
            return self.llm
        if self.model_name is not None:
            return get_chat_model(self.model_name)
        raise ValueError("Either llm or model_name must be provided")

    def _create_prompt_with_attachment(
        self,
        query: str,
        context: str | None = None,
        file_attachment: FileAttachmentSource | None = None,
    ) -> list[SystemMessage | HumanMessage]:
        """Build the system and user messages, with the attachment in the user message."""
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=self.system_instructions)
        ]

        if file_attachment is not None:
            attachment = file_attachment.to_attachment(self.documents_dir)

            user_content: list[Any] = []

            query_text = f"Query: {query}"
            if context:
                query_text += f"\n\nContext: {context}"
            user_content.append({"type": "text", "text": query_text})

            if attachment.source_type == "base64":
                if attachment.mime_type == "application/pdf":
                    user_content.append(
                        {
                            "type": "file",
                            "source_type": "base64",
                            "data": attachment.data,
                            "mime_type": attachment.mime_type,
                        }
                    )
                else:
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{attachment.mime_type};base64,{attachment.data}"
                            },
                        }
                    )
            elif attachment.source_type == "text":
                text_content = attachment.text_content or ""
                if attachment.filename:
                    text_content = f"File: {attachment.filename}\n\n{text_content}"
                user_content.append({"type": "text", "text": text_content})
            else:
                raise ValueError(
                    f"Unsupported attachment source type. Got source_type: {attachment.source_type}"
                )

            messages.append(HumanMessage(content=user_content))
        else:
            user_message = f"Query: {query}"
            if context:
                user_message += f"\n\nContext: {context}"
            messages.append(HumanMessage(content=user_message))

        return messages

    @override
    def _run(
        self,
        query: str,
        context: str | None = None,
        file_attachment: FileAttachmentSource | None = None,
        **kwargs: Any,
    ) -> Any:
        llm = self._get_llm()

        if file_attachment is not None:
            messages = self._create_prompt_with_attachment(query, context, file_attachment)

            if self.structured_output_schema is not None:
                result = llm.with_structured_output(
                    self.structured_output_schema.model_json_schema(),
                    strict=gym_settings_core.UIPATH_AGENTS_GYM_LLM_STRICT_MODE,
                ).invoke(messages)  # type: ignore
                return result
            else:
                response = llm.invoke(messages)
                return PlatformAnalyzeAttachmentOutput(response=str(response.content))
        else:
            raise ValueError("File attachment is required")

    @override
    async def _arun(
        self,
        query: str,
        context: str | None = None,
        file_attachment: FileAttachmentSource | None = None,
        **kwargs: Any,
    ) -> Any:
        llm = self._get_llm()

        if file_attachment is not None:
            messages = self._create_prompt_with_attachment(query, context, file_attachment)

            if self.structured_output_schema is not None:
                result = await llm.with_structured_output(
                    self.structured_output_schema.model_json_schema(),
                    strict=gym_settings_core.UIPATH_AGENTS_GYM_LLM_STRICT_MODE,
                ).ainvoke(messages)
                return result
            else:
                response = await llm.ainvoke(messages)
                return PlatformAnalyzeAttachmentOutput(response=str(response.content))
        else:
            raise ValueError("File attachment is required")
