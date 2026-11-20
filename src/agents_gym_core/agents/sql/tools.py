"""SQL Agent tools using LangChain's SQLDatabaseToolkit."""

import ast
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, event
from sqlalchemy.pool import NullPool
from uipath_langchain.agent.tools.tool_node import ToolWrapperMixin

from agents_gym_core.models.chat import OpenAIModels, get_chat_model

QUERY_CHECK_SYSTEM = """You are a SQL expert with a strong attention to detail.
    Double check the SQLite query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query."""

QUERY_RESULT_CHECK_SYSTEM = """You are grading the result of a SQL query from a DB.
- Check that the result is not empty.
- If it is empty, instruct the system to re-try!"""


class QueryCheckerTool(BaseTool):
    """Tool to check if a SQL query is correct before executing it."""

    name: str = "check_query_tool"
    description: str = "Use this tool to check if your SQL query is correct before executing it."
    query_checker: Runnable | None = None

    def __init__(self, llm: BaseChatModel, **kwargs: Any):
        super().__init__(**kwargs)
        query_check_prompt = ChatPromptTemplate.from_messages(
            [("system", QUERY_CHECK_SYSTEM), ("user", "{query}")]
        )
        self.query_checker = query_check_prompt | llm

    def _run(self, query: str) -> str:
        if self.query_checker:
            result = self.query_checker.invoke({"query": query})
            return result.content if hasattr(result, "content") else str(result)
        return ""

    async def _arun(self, query: str) -> str:
        if self.query_checker:
            result = await self.query_checker.ainvoke({"query": query})
            return result.content if hasattr(result, "content") else str(result)
        return ""


# Deterministic checks of a query result against the question it should answer. They add a
# note to the checker's reply only when a result is clearly wrong for the question; any other
# result gets the checker's reply unchanged.
_REQUEST_LEAD = re.compile(
    r"^\s*(please\s+)?(find|get|retrieve|list|show|return|give|calculate|compute|determine|"
    r"identify|select|display|count|what\s+(is|are|was|were)|which)\s+(me\s+)?"
    r"(the\s+|all\s+(the\s+)?)?",
    re.IGNORECASE,
)
_REQUEST_END = re.compile(
    r"\s+(of|for|in|from|with|that|whose|who|where|which|by|on|per|among|between|across)\s+",
    re.IGNORECASE,
)
_NUMBER_REQUEST = re.compile(
    r"\b(total|number|count|sum|average|amount|price|quantity|cost|duration|length)\b",
    re.IGNORECASE,
)
_EMPTY_RESULTS = {"", "[]", "()", "[()]", "none", "null", "nan"}
_FAILED_QUERY = re.compile(
    r"^\s*error\b|\boperationalerror\b|\bno such (table|column)\b|\bsyntax error\b", re.IGNORECASE
)
# A question asks for several rows when it says "all" or names a plural (in the item it asks
# for, or in the "of the ..." phrase that follows it).
_SEVERAL = re.compile(
    r"\b(tables|indexes|columns|relationships|employees|customers|tracks|albums|artists|genres|"
    r"playlists|invoices)\b",
    re.IGNORECASE,
)


def requested_item(question: str) -> str:
    """What a question asks for: "Find the invoice total for ..." gives "invoice total"."""
    return _REQUEST_END.split(_REQUEST_LEAD.sub("", question.strip()), 1)[0]


def _asks_for_several(question: str, item: str) -> bool:
    of_the = re.search(r"\bof\s+the\s+(\w+)", question, re.IGNORECASE)
    return bool(
        re.search(r"\ball\b", question, re.IGNORECASE)
        or re.search(r"\b(per|each|every|by)\b", question, re.IGNORECASE)  # "total per country"
        or _SEVERAL.search(item)
        or (of_the and _SEVERAL.fullmatch(of_the.group(1)))
    )


def _rows(result: str) -> list[tuple] | None:
    """The rows of a raw SQL result such as "[('AC/DC',), ('Accept',)]"; None for other text."""
    if not (result.startswith("[(") and result.endswith(")]")):
        return None
    try:
        rows = ast.literal_eval(result)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return rows if isinstance(rows, list) and all(isinstance(r, tuple) for r in rows) else None


def result_problems(question: str, query_result: str) -> list[str]:
    """Problems of a query result that are certain without judging its content; [] if none."""
    result = query_result.strip()
    if result.lower() in _EMPTY_RESULTS or not re.search(r"\w", result):
        return ["Check: the result is empty. Re-check the query before answering."]
    if _FAILED_QUERY.search(result):
        return ["Check: the query failed. Fix it before answering."]
    item = requested_item(question)
    problems = []
    asks_number = _NUMBER_REQUEST.search(item) or re.match(
        r"\s*how\s+(many|much)\b", question, re.IGNORECASE
    )
    if asks_number and not re.search(r"\d", result):
        problems.append(f"Check: the question asks for the {item}, but the result has no number.")
    if re.search(r"\be-?mail", item, re.IGNORECASE) and "@" not in result:
        problems.append("Check: the question asks for an email, but the result has no address.")
    rows = _rows(result)
    if rows is not None and len(rows) > 1 and not _asks_for_several(question, item):
        problems.append(
            f"Check: the question asks for one {item}, but the result has {len(rows)} rows."
        )
    return problems


class _QuestionState(BaseModel):
    """The part of the agent's state the result check reads: the SQL agent's ``query`` input."""

    query: str = ""


def _add_problems(message: Any, call: ToolCall, state: _QuestionState) -> Any:
    problems = result_problems(state.query, str(call["args"].get("query_result", "")))
    if problems and isinstance(message, ToolMessage) and isinstance(message.content, str):
        message.content = "\n".join([message.content, *problems])
    return message


def _check_against_question(tool: BaseTool, call: ToolCall, state: _QuestionState) -> Any:
    """Run the tool exactly as the tool node would without a wrapper, then add any problems."""
    return _add_problems(tool.invoke(call), call, state)


async def _acheck_against_question(tool: BaseTool, call: ToolCall, state: _QuestionState) -> Any:
    return _add_problems(await tool.ainvoke(call), call, state)


class QueryResultCheckerTool(BaseTool, ToolWrapperMixin):
    """Tool to check the result of a SQL query.

    An LLM checks that the result is not empty. In the agent loop, the tool node also passes
    the agent's question (``query``) to :func:`result_problems`, whose notes are appended to
    the reply. What the model sees of the tool (name, description, arguments) is unchanged.
    """

    name: str = "check_result_tool"
    description: str = "Use this tool to check the result of your SQL query."
    result_check: Runnable | None = None

    def __init__(self, llm: BaseChatModel, **kwargs: Any):
        super().__init__(**kwargs)
        result_check_prompt = ChatPromptTemplate.from_messages(
            [("system", QUERY_RESULT_CHECK_SYSTEM), ("user", "{query_result}")]
        )
        self.result_check = result_check_prompt | llm
        self.set_tool_wrappers(wrapper=_check_against_question, awrapper=_acheck_against_question)

    def _run(self, query_result: str) -> str:
        if self.result_check:
            result = self.result_check.invoke({"query_result": query_result})
            return result.content if hasattr(result, "content") else str(result)
        return ""

    async def _arun(self, query_result: str) -> str:
        if self.result_check:
            result = await self.result_check.ainvoke({"query_result": query_result})
            return result.content if hasattr(result, "content") else str(result)
        return ""


def _read_only_uri(db_path: str) -> str:
    """SQLAlchemy URI that opens the database read-only, so agent queries cannot modify it."""
    return f"sqlite:///{Path(db_path).resolve().as_uri()}?mode=ro&uri=true"


def _read_only_db(db_path: str) -> SQLDatabase:
    """The database opened read-only and with ATTACH disabled.

    Read-only mode covers only the main database: ``ATTACH`` (and ``VACUUM INTO``, which
    attaches its target) would let a query read, change or create any other SQLite file.
    """
    engine = create_engine(_read_only_uri(db_path), poolclass=NullPool)

    @event.listens_for(engine, "connect")
    def _no_attach(dbapi_connection: sqlite3.Connection, _record: object) -> None:
        dbapi_connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)

    return SQLDatabase(engine)


class SQLTools(BaseModel):
    """SQL Tools provider that can be instantiated by Hydra config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    db_path: str | None = None
    llm: BaseChatModel | None = None

    def __call__(self) -> Sequence[BaseTool]:
        """Get SQL database tools when called as a function.

        Returns:
            List of SQL tools including:
            - sql_db_list_tables: List all tables in the database
            - sql_db_schema: Get schema for specified tables
            - sql_db_query_checker: Check if query is correct
            - sql_db_query: Execute SQL query
            - check_result_tool: Check query result
        """
        if not self.db_path:
            raise ValueError("db_path must be provided")
        if not self.llm:
            raise ValueError("llm must be provided")

        db_path_str = str(Path(self.db_path).resolve())
        db = _read_only_db(db_path_str)
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        langchain_tools = toolkit.get_tools()

        # Get the standard LangChain SQL tools
        tools = [
            tool
            for tool in langchain_tools
            if tool.name
            in [
                "sql_db_list_tables",
                "sql_db_schema",
                "sql_db_query_checker",
                "sql_db_query",
            ]
        ]

        # Add custom result checker tool
        tools.append(QueryResultCheckerTool(llm=self.llm))

        return tools


def get_sql_tools(
    db_path: str | Path,
    llm: BaseChatModel,
) -> Sequence[BaseTool]:
    """Get SQL database tools including LangChain toolkit and custom tools.

    Args:
        db_path: Path to the SQLite database file
        llm: Language model for query checking tools

    Returns:
        List of SQL tools including:
        - sql_db_list_tables: List all tables in the database
        - sql_db_schema: Get schema for specified tables
        - sql_db_query_checker: Check if query is correct
        - sql_db_query: Execute SQL query
        - check_result_tool: Check query result
    """
    tools_provider = SQLTools(db_path=str(db_path), llm=llm)
    return tools_provider()


def _get_default_db_path() -> str:
    """Get the default database path."""
    module_dir = Path(__file__).parent.resolve()
    return str(module_dir / "documents" / "sql.db")


def _get_default_llm() -> BaseChatModel:
    """LLM behind the SQL query-checker tool (``gpt-4o-2024-11-20``), on the configured backend."""
    return get_chat_model(OpenAIModels.gpt_4o_2024_11_20.name)


def _get_db_and_toolkit(db_path: str | None = None) -> tuple[SQLDatabase, SQLDatabaseToolkit]:
    """Get database and toolkit instances."""
    if db_path is None:
        db_path = _get_default_db_path()

    db = _read_only_db(str(db_path))
    llm = _get_default_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return db, toolkit


def create_sql_db_list_tables_tool(db_path: str | None = None) -> BaseTool:
    """Create sql_db_list_tables tool for use in agent.yaml."""
    _, toolkit = _get_db_and_toolkit(db_path)
    tools = toolkit.get_tools()
    return next(t for t in tools if t.name == "sql_db_list_tables")


def create_sql_db_schema_tool(db_path: str | None = None) -> BaseTool:
    """Create sql_db_schema tool for use in agent.yaml."""
    _, toolkit = _get_db_and_toolkit(db_path)
    tools = toolkit.get_tools()
    return next(t for t in tools if t.name == "sql_db_schema")


def create_sql_db_query_checker_tool(db_path: str | None = None) -> BaseTool:
    """Create sql_db_query_checker tool for use in agent.yaml."""
    _, toolkit = _get_db_and_toolkit(db_path)
    tools = toolkit.get_tools()
    return next(t for t in tools if t.name == "sql_db_query_checker")


def create_sql_db_query_tool(db_path: str | None = None) -> BaseTool:
    """Create sql_db_query tool for use in agent.yaml."""
    _, toolkit = _get_db_and_toolkit(db_path)
    tools = toolkit.get_tools()
    return next(t for t in tools if t.name == "sql_db_query")


def create_check_result_tool(db_path: str | None = None) -> BaseTool:
    """Create check_result_tool for use in agent.yaml."""
    llm = _get_default_llm()
    return QueryResultCheckerTool(llm=llm)
