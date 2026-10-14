"""Offline integration coverage for conversation-owned Excel artifacts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest

import src.gaia_agent.tools.file_generation.generation as generation_module
from src.gaia_agent.runtime.contracts import (
    ChatMessage,
    ContextInfo,
    ModelTurn,
    ToolCall,
    ToolDefinition,
)
from src.gaia_agent.runtime.service import build_gaia_agent
from src.gaia_agent.tools.query.execution import ExportQueryResult
from src.gaia_agent.tools.query.plan import ValidatedQueryPlan
from src.gaia_agent.web.chat_service import (
    GaiaChatResponse,
    GaiaChatService,
    UnknownConversationError,
    UnknownGeneratedFileError,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ExportClient:
    """Request marked dataset exports per user turn and capture the observation."""

    def __init__(
        self,
        barrier: threading.Barrier | None = None,
        *,
        layout: str = "single_workbook",
        filename: str | None = None,
    ) -> None:
        self.results: dict[str, dict[str, object]] = {}
        self._barrier = barrier
        self._layout = layout
        self._filename = filename
        self._lock = threading.Lock()

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        max_completion_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> ModelTurn:
        marker = next(message.content for message in reversed(messages) if message.role == "user")
        assert isinstance(marker, str)
        if messages[-1].role == "user":
            assert any(tool.name == "generate_excel" for tool in tools)
            if self._barrier is not None:
                # Both per-turn runtimes exist before either invokes its tool.
                # A shared mutable destination would route one user's file
                # into the other conversation's directory.
                self._barrier.wait(timeout=5)
            return ModelTurn(tool_calls=(ToolCall(
                f"file_{marker}",
                "generate_excel",
                {
                    "queries": [{
                        "scope": {"dataset": dataset},
                        "query": {
                            "filters": {"occurrenceID": {"exact_any": [marker]}},
                            "column_mode": "selected",
                            "return_columns": ["catalogNumber"],
                        },
                        "result": {"distinct_by": [], "order_by": []},
                    } for dataset in ("Birds", "Fish")],
                    "limit": None,
                    "limit_scope": "per_dataset",
                    "layout": self._layout,
                    **({"filename": self._filename} if self._filename is not None else {}),
                },
                purpose="Create the requested occurrence workbook.",
            ),))
        assert messages[-1].role == "tool"
        assert messages[-1].content is not None
        payload = json.loads(messages[-1].content)
        assert payload["ok"] is True
        with self._lock:
            self.results[marker] = payload["result"]
        return ModelTurn(content="\n".join(
            file["filename"] for file in payload["result"]["files"]
        ))

    async def context_info(
        self, *, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition],
        enable_thinking: bool | None = None,
    ) -> ContextInfo:
        return ContextInfo(prompt_tokens=100, context_limit_tokens=131_072)

    async def count_text_tokens(self, text: str) -> int:
        return len(text)


async def _stream_occurrence_plans(
    plans: Sequence[ValidatedQueryPlan],
    *,
    on_batch: Callable[[str, tuple[str, ...], tuple[tuple[str, ...], ...]], Awaitable[None]],
    on_dataset_complete: Callable[[str], Awaitable[None]],
    row_limit: int | None = None,
    limit_scope: str = "total",
    **_arguments: object,
) -> ExportQueryResult:
    assert tuple(plan.dataset_keys for plan in plans) == (("birds",), ("fish",))
    assert row_limit is None
    assert limit_scope == "per_dataset"
    for plan in plans:
        assert plan.return_columns == ("catalogNumber",)
        marker = plan.filters[0].exact_any[0]
        dataset_key = plan.dataset_keys[0]
        headers = ("occurrenceID", "catalogNumber")
        await on_batch(dataset_key, headers, ())
        await on_batch(dataset_key, headers, ((marker, "0001"),))
        await on_dataset_complete(dataset_key)
    return ExportQueryResult(
        selected_dataset_keys=("birds", "fish"),
        compatible_dataset_keys=("birds", "fish"),
        queried_dataset_keys=("birds", "fish"),
        skipped_datasets=(),
        dataset_row_counts=(("birds", 1), ("fish", 1)),
        limit=row_limit,
        database_truncated=False,
    )


@pytest.fixture
def stub_occurrence_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generation_module, "stream_query_plans", _stream_occurrence_plans)


def _chat_service(
    repository_root: Path,
    client: ExportClient,
    *,
    trace_root: Path | None = None,
) -> GaiaChatService:
    agent = build_gaia_agent(
        repository_root,
        client,
        system_prompt_path=REPOSITORY_ROOT / "src/gaia_agent/runtime/system_prompt.md",
    )
    return GaiaChatService(agent, repository_root=repository_root, trace_root=trace_root)


def _assert_session_export(
    response: GaiaChatResponse,
    metadata: dict[str, object],
    *,
    marker: str,
    username: str,
    repository_root: Path,
    trace_root: Path,
) -> list[Path]:
    files = metadata["files"]
    paths = [attachment.path for attachment in response.attachments]
    assert all("path" not in file for file in files)
    assert str(repository_root) not in json.dumps(metadata)
    assert str(trace_root) not in json.dumps(response.payload())
    assert len(paths) == (1 if metadata["layout"] == "single_workbook" else 2)
    assert [file.file_id for file in response.attachments] == [
        file["file_id"] for file in files
    ]
    assert response.payload()["message"]["attachments"] == [
        {
            "file_id": file["file_id"],
            "filename": file["filename"],
            "size_bytes": path.stat().st_size,
            "media_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        for file, path in zip(files, paths, strict=True)
    ]
    assert metadata["record_count"] == 2
    assert response.content == "\n".join(path.name for path in paths)
    assert len({path.parent for path in paths}) == 1
    assert paths[0].parent.name == "exports"
    session_directory = paths[0].parent.parent
    assert session_directory.parent == trace_root
    assert re.fullmatch(rf"gaia_chat_{username}_\d{{8}}T\d{{6}}_\d{{6}}Z", session_directory.name)
    trace_path = session_directory / f"interaction_{response.interaction:04d}.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["run"]["status"] == "completed"
    assert trace["run"]["invocation"]["username"] == username
    assert trace["user_prompt"] == marker
    assert len(trace["tools_requested"]) == 1
    logged_tool = trace["tools_requested"][0]
    assert logged_tool["name"] == "generate_excel"
    assert logged_tool["returned_result"] == {
        "ok": True,
        "result": metadata,
    }
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    dataset_keys = []
    for file, path in zip(files, paths, strict=True):
        assert path.is_file() and path.name == file["filename"]
        assert file["record_count"] == len(file["datasets"])
        with ZipFile(path) as archive:
            for index, dataset in enumerate(file["datasets"], start=1):
                dataset_keys.append(dataset["dataset_key"])
                assert dataset["row_count"] == 1
                sheet = ET.fromstring(archive.read(f"xl/worksheets/sheet{index}.xml"))
                rows = [
                    tuple("".join(node.text or "" for node in cell.findall(".//s:t", namespace))
                          for cell in row.findall("s:c", namespace))
                    for row in sheet.findall("s:sheetData/s:row", namespace)
                ]
                assert rows == [("occurrenceID", "catalogNumber"), (marker, "0001")]
    assert dataset_keys == ["birds", "fish"]
    return paths


@pytest.mark.parametrize("layout", ["single_workbook", "separate_files"])
def test_exports_follow_user_sessions_and_reuse_the_followup_directory(
    tmp_path: Path,
    stub_occurrence_stream: None,
    layout: str,
) -> None:
    repository_root = tmp_path / "repository"
    client = ExportClient(layout=layout)
    service = _chat_service(repository_root, client)
    first = service.chat("researcher_first", username=" Field Researcher #42 ")
    other = service.chat("curator_first", username="Collections Curator")
    followup = service.chat(
        "researcher_followup",
        username="field-researcher 42",
        conversation_id=first.conversation_id,
    )
    separate = service.chat("researcher_new_session", username="Field Researcher #42")
    path_groups = [
        _assert_session_export(
            response, client.results[marker], marker=marker, username=username,
            repository_root=repository_root, trace_root=repository_root / "logs",
        )
        for response, marker, username in (
            (first, "researcher_first", "field_researcher_42"),
            (other, "curator_first", "collections_curator"),
            (followup, "researcher_followup", "field_researcher_42"),
            (separate, "researcher_new_session", "field_researcher_42"),
        )
    ]
    file_count = 1 if layout == "single_workbook" else 2
    assert [[path.name for path in group] for group in path_groups] == [
        [f"{username}_{number}.xlsx" for number in range(start, start + file_count)]
        for username, start in (
            ("field_researcher_42", 1),
            ("collections_curator", 1),
            ("field_researcher_42", file_count + 1),
            ("field_researcher_42", 1),
        )
    ]
    assert followup.conversation_id == first.conversation_id
    assert followup.interaction == 2
    assert separate.conversation_id != first.conversation_id
    paths = [path for group in path_groups for path in group]
    assert path_groups[0][0].parent == path_groups[2][0].parent
    assert len({path.parent for path in paths}) == 3
    assert len(set(paths)) == (4 if layout == "single_workbook" else 8)
    assert set(repository_root.rglob("*.xlsx")) == set(paths)
    assert set(paths[0].parent.glob("*.xlsx")) == set(path_groups[0] + path_groups[2])
    for response in (first, followup):
        for attachment in response.attachments:
            metadata, stream = service.open_file(
                username="field_researcher_42",
                conversation_id=response.conversation_id,
                file_id=attachment.file_id,
            )
            with stream:
                assert stream.read() == metadata.path.read_bytes()
            with pytest.raises(UnknownConversationError):
                service.open_file(
                    username="collections_curator",
                    conversation_id=response.conversation_id,
                    file_id=attachment.file_id,
                )
            with pytest.raises(UnknownGeneratedFileError):
                service.open_file(
                    username="field_researcher_42",
                    conversation_id=separate.conversation_id,
                    file_id=attachment.file_id,
                )


@pytest.mark.parametrize("layout", ["single_workbook", "separate_files"])
def test_concurrent_users_keep_exports_in_their_custom_trace_root(
    tmp_path: Path,
    stub_occurrence_stream: None,
    layout: str,
) -> None:
    repository_root = tmp_path / "repository"
    trace_root = tmp_path / "custom_traces"
    client = ExportClient(threading.Barrier(2), layout=layout)
    service = _chat_service(repository_root, client, trace_root=trace_root)
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending = {
            username: workers.submit(service.chat, f"parallel_{username}", username=username)
            for username in ("alice", "bob")
        }
        responses = {username: future.result(timeout=10) for username, future in pending.items()}
    path_groups = [
        _assert_session_export(
            response, client.results[f"parallel_{username}"], marker=f"parallel_{username}",
            username=username, repository_root=repository_root, trace_root=trace_root,
        )
        for username, response in responses.items()
    ]
    for username, group in zip(responses, path_groups, strict=True):
        assert [path.name for path in group] == [
            f"{username}_{number}.xlsx" for number in range(1, len(group) + 1)
        ]
    assert responses["alice"].conversation_id != responses["bob"].conversation_id
    assert path_groups[0][0].parent != path_groups[1][0].parent
    paths = [path for group in path_groups for path in group]
    assert set(trace_root.rglob("*.xlsx")) == set(paths)
    assert not list(repository_root.rglob("*.xlsx"))


@pytest.mark.parametrize(
    ("layout", "expected_filenames"),
    [
        ("single_workbook", ["Field notes – 雪.xlsx"]),
        ("separate_files", ["Field notes – 雪-birds.xlsx", "Field notes – 雪-fish.xlsx"]),
    ],
)
def test_exports_use_requested_filenames_and_preserve_existing_downloads(
    tmp_path: Path,
    stub_occurrence_stream: None,
    layout: str,
    expected_filenames: list[str],
) -> None:
    repository_root = tmp_path / "repository"
    trace_root = tmp_path / "custom_traces"
    client = ExportClient(layout=layout, filename="Field notes – 雪")
    service = _chat_service(repository_root, client, trace_root=trace_root)
    first = service.chat("requested_name", username="researcher")
    first_paths = _assert_session_export(
        first, client.results["requested_name"], marker="requested_name", username="researcher",
        repository_root=repository_root, trace_root=trace_root,
    )
    original_bytes = {path: path.read_bytes() for path in first_paths}
    assert [path.name for path in first_paths] == expected_filenames

    followup = service.chat(
        "same_requested_name", username="researcher", conversation_id=first.conversation_id,
    )
    followup_paths = _assert_session_export(
        followup, client.results["same_requested_name"], marker="same_requested_name",
        username="researcher", repository_root=repository_root, trace_root=trace_root,
    )
    assert [path.name for path in followup_paths] == [
        f"{filename[:-5]}-2.xlsx" for filename in expected_filenames
    ]
    assert {path.parent for path in first_paths + followup_paths} == {first_paths[0].parent}
    assert not {file.file_id for file in first.attachments} & {
        file.file_id for file in followup.attachments
    }
    for attachment, expected_path in zip(first.attachments, first_paths, strict=True):
        metadata, stream = service.open_file(
            username="researcher",
            conversation_id=first.conversation_id,
            file_id=attachment.file_id,
        )
        assert metadata.path == expected_path
        with stream:
            assert stream.read() == original_bytes[expected_path]
    assert set(trace_root.rglob("*.xlsx")) == set(first_paths + followup_paths)
