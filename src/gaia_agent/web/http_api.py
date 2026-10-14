"""Serve Gaia's local chat API with JSON responses or NDJSON progress."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import BinaryIO, cast
from urllib.parse import parse_qsl, quote, urlsplit

from src.logger import normalize_chat_username
from src.logger.gaia_logger import GaiaLoggerError

from src.gaia_agent.runtime.defaults import (
    DEFAULT_GAIA_HTTP_HOST,
    DEFAULT_GAIA_HTTP_PORT,
)
from src.gaia_agent.runtime.contracts import (
    AgentCancellation,
    AgentCancelledError,
    AgentRuntimeError,
    AgentProgressEvent,
    ContextCompactionError,
    ToolAttemptsExhaustedError,
    ToolCallLimitExceededError,
)
from src.gaia_agent.runtime.openai_client import OpenAIClientError
from src.gaia_agent.tools.concurrency import WorkloadBusyError
from src.gaia_agent.tools.file_generation.artifacts import GeneratedFile
from src.gaia_agent.web.activity_summaries import summarize_gaia_tool_result
from src.gaia_agent.web.chat_service import (
    ConversationNotRunningError,
    GaiaChatResponse,
    GaiaChatService,
    UnknownConversationError,
    UnknownGeneratedFileError,
    UnknownInteractionError,
    is_valid_conversation_id,
    require_positive_integer,
)


DEFAULT_MAX_HTTP_REQUEST_BYTES = 32_768
DEFAULT_MAX_CHAT_MESSAGE_CHARACTERS = 8_000
DEFAULT_MAX_CHAT_MESSAGE_BYTES = 24_576
GAIA_HTTP_ALLOWED_ORIGINS_ENV = "GAIA_HTTP_ALLOWED_ORIGINS"

DEFAULT_CORS_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://beaty-biodiversity-query.noslopplsthx.chatgpt.site",
)

_CHAT_PATH = "/api/chat"
_FEEDBACK_PATH = "/api/feedback"
_FILES_PATH = "/api/files"
_HEALTH_PATH = "/api/health"
_CHAT_STREAM_MEDIA_TYPE = "application/x-ndjson"
_ROUTE_METHODS = {
    _CHAT_PATH: ("DELETE", "POST"),
    _FEEDBACK_PATH: ("POST",),
    _FILES_PATH: ("GET",),
    _HEALTH_PATH: ("GET",),
}
_MAX_PUBLIC_ACTIVITIES_PER_INTERACTION = 5
_FILE_CHUNK_BYTES = 65_536
_PREFLIGHT_VARY = (
    "Origin, Access-Control-Request-Method, Access-Control-Request-Headers, "
    "Access-Control-Request-Private-Network"
)


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------


def validate_cors_allowed_origins(
    origins: Iterable[str],
) -> tuple[str, ...]:
    """Return unique exact HTTP origins or reject unsafe wildcard values."""
    if isinstance(origins, (str, bytes, bytearray)):
        raise TypeError("allowed origins must be an iterable of origin strings")
    validated: list[str] = []
    for origin in origins:
        if not isinstance(origin, str) or not origin:
            raise ValueError("allowed origins must contain non-empty strings")
        if origin == "*":
            raise ValueError("wildcard CORS origins are not supported")
        parsed = urlsplit(origin)
        try:
            parsed.port
        except ValueError as error:
            raise ValueError(f"invalid CORS origin: {origin!r}") from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or origin != f"{parsed.scheme}://{parsed.netloc}"
        ):
            raise ValueError(f"invalid CORS origin: {origin!r}")
        if origin not in validated:
            validated.append(origin)
    return tuple(validated)


def load_cors_allowed_origins(
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Load an exact comma-separated allowlist from the process environment."""
    active_environment = os.environ if environment is None else environment
    configured = active_environment.get(GAIA_HTTP_ALLOWED_ORIGINS_ENV)
    if configured is None:
        return DEFAULT_CORS_ALLOWED_ORIGINS
    return validate_cors_allowed_origins(
        origin.strip()
        for origin in configured.split(",")
        if origin.strip()
    )


# ---------------------------------------------------------------------------
# HTTP boundary
# ---------------------------------------------------------------------------


class _HttpProblem(RuntimeError):
    def __init__(
        self,
        status: HTTPStatus,
        code: str,
        public_message: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(public_message)
        self.status = status
        self.code = code
        self.public_message = public_message
        self.headers = headers


def _problem_from_error(
    error: Exception, *, feedback: bool = False,
) -> _HttpProblem:
    """Map request and service failures to the public error contract."""
    if isinstance(error, _HttpProblem):
        return error
    if isinstance(error, WorkloadBusyError) or (
        isinstance(error, ToolAttemptsExhaustedError)
        and error.failed_result.error_code == "server_busy"
    ):
        return _HttpProblem(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "server_busy",
            "Gaia is busy. Please try again shortly.",
        )
    if feedback and isinstance(error, (TypeError, ValueError)):
        return _HttpProblem(
            HTTPStatus.BAD_REQUEST,
            "invalid_feedback",
            "Feedback has an invalid format.",
        )
    if isinstance(error, UnknownInteractionError):
        return _HttpProblem(
            HTTPStatus.NOT_FOUND,
            "unknown_interaction",
            "The completed interaction does not exist or is unavailable.",
        )
    if isinstance(error, UnknownGeneratedFileError):
        return _HttpProblem(
            HTTPStatus.NOT_FOUND,
            "unknown_file",
            "The generated file does not exist or is no longer available.",
        )
    if isinstance(error, ConversationNotRunningError):
        return _HttpProblem(
            HTTPStatus.CONFLICT,
            "no_active_request",
            "This conversation has no active request to cancel.",
        )
    if isinstance(error, UnknownConversationError):
        return _HttpProblem(
            HTTPStatus.NOT_FOUND,
            "unknown_conversation",
            "The conversation does not exist or is no longer available.",
        )
    if isinstance(error, ToolCallLimitExceededError):
        return _HttpProblem(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "request_too_complex",
            "Gaia could not finish this request in one interaction. "
            "Please narrow the request or continue in a new message.",
        )
    if isinstance(error, ToolAttemptsExhaustedError):
        if error.failed_result.error_code == "query_unavailable":
            return _HttpProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "catalogue_unavailable",
                "The biodiversity catalogue is temporarily unavailable. "
                "Please try again later.",
            )
        if error.failed_result.error_code in {
            "invalid_arguments",
            "invalid_query_plan",
        }:
            return _HttpProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "invalid_search",
                "Gaia could not complete this search. "
                "Please check or clarify the requested collection, filters, "
                "and fields, then try again.",
            )
        return _HttpProblem(
            HTTPStatus.BAD_GATEWAY,
            "request_failed",
            "Gaia could not complete this request. Please try again.",
        )
    if isinstance(error, ContextCompactionError):
        return _HttpProblem(
            HTTPStatus.BAD_GATEWAY,
            "context_compaction_failed",
            "Gaia could not compact the earlier conversation. "
            "This request could not continue.",
        )
    if isinstance(error, AgentRuntimeError):
        return _HttpProblem(
            HTTPStatus.BAD_GATEWAY,
            "response_unavailable",
            "Gaia could not complete this response. Please try again.",
        )
    if isinstance(error, OpenAIClientError):
        return _HttpProblem(
            HTTPStatus.BAD_GATEWAY,
            "service_unavailable",
            "Gaia is temporarily unavailable. Please try again later.",
        )
    if isinstance(error, GaiaLoggerError):
        return _HttpProblem(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "internal_error",
            "Gaia could not complete this interaction.",
        )
    return _HttpProblem(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "internal_error",
        "Gaia could not complete this interaction.",
    )


class GaiaHttpServer(ThreadingHTTPServer):
    """Bind Gaia's chat service and exact browser-origin policy."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        chat_service: GaiaChatService,
        *,
        allowed_origins: Iterable[str] = DEFAULT_CORS_ALLOWED_ORIGINS,
        max_request_bytes: int = DEFAULT_MAX_HTTP_REQUEST_BYTES,
        max_message_characters: int = DEFAULT_MAX_CHAT_MESSAGE_CHARACTERS,
        max_message_bytes: int = DEFAULT_MAX_CHAT_MESSAGE_BYTES,
    ) -> None:
        require_positive_integer(max_request_bytes, "max request bytes")
        require_positive_integer(
            max_message_characters,
            "max message characters",
        )
        require_positive_integer(max_message_bytes, "max message bytes")
        self.chat_service = chat_service
        self.allowed_origins = frozenset(
            validate_cors_allowed_origins(allowed_origins)
        )
        self.max_request_bytes = max_request_bytes
        self.max_message_characters = max_message_characters
        self.max_message_bytes = max_message_bytes
        super().__init__(server_address, GaiaHttpRequestHandler)


@dataclass(slots=True)
class _ChatStream:
    """Keep connection, cancellation, and activity state for one streamed turn."""

    write_event: Callable[[Mapping[str, object]], bool]
    conversation_id: str | None
    cancellation: AgentCancellation = field(default_factory=AgentCancellation)
    is_connected: bool = True
    terminal_emitted: bool = False
    emitted_activities: int = 0

    def emit(self, payload: Mapping[str, object]) -> bool:
        if self.is_connected:
            self.is_connected = self.write_event(payload)
        return self.is_connected

    def report_progress(self, event: AgentProgressEvent) -> None:
        payload: dict[str, object] = {"type": "progress", "phase": "thinking"}
        if event.kind == "tool_batch_started":
            if "generate_excel" in event.tool_names:
                payload["phase"] = "preparing_file"
            elif any(
                name in {"query_occurrences", "query_statistics"}
                for name in event.tool_names
            ):
                payload["phase"] = "searching"
            else:
                payload["phase"] = "reviewing"
        elif event.kind == "tool_batch_completed":
            payload["phase"] = "activity"
            payload["activities"] = self._activities(event)
        elif event.kind == "response_ready":
            payload["phase"] = "preparing_answer"
        elif event.kind in {"context_compacting", "context_compacted"}:
            payload["phase"] = event.kind
        if not self.emit(payload):
            self.cancellation.cancel()

    def _activities(self, event: AgentProgressEvent) -> list[dict[str, object]]:
        remaining = max(
            0, _MAX_PUBLIC_ACTIVITIES_PER_INTERACTION - self.emitted_activities,
        )
        activities = [
            {
                "id": f"activity-{self.emitted_activities + index}",
                **summarize_gaia_tool_result(result),
            }
            for index, result in enumerate(event.tool_results[:remaining], start=1)
        ]
        self.emitted_activities += len(activities)
        return activities

    def deliver_conversation_id(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        if not self.emit({"type": "conversation", "conversation_id": conversation_id}):
            self.cancellation.cancel()

    def deliver_response(self, response: GaiaChatResponse) -> None:
        self.terminal_emitted = self.emit(
            {"type": "response", "data": response.payload()}
        )
        if not self.terminal_emitted:
            self.cancellation.cancel()
            raise AgentCancelledError(
                "Stopped thinking",
                transcript=(),
                tool_results=(),
                model_turns=response.model_turns,
            )

    def deliver_cancellation(self) -> None:
        if not self.terminal_emitted and self.is_connected:
            self.emit(
                {
                    "type": "cancelled",
                    "conversation_id": self.conversation_id,
                    "message": "Stopped thinking",
                }
            )

    def deliver_error(self, error: Exception) -> None:
        problem = _problem_from_error(error)
        self.emit(
            {
                "type": "error",
                "status": problem.status.value,
                "error": {
                    "code": problem.code,
                    "message": problem.public_message,
                },
            }
        )


class GaiaHttpRequestHandler(BaseHTTPRequestHandler):
    """Handle Gaia's small JSON HTTP surface without exposing internals."""

    server_version = "GaiaHTTP/1"
    sys_version = ""

    def _handle_request(self) -> None:
        origin: str | None = None
        stream_request: tuple[str, str, str | None] | None = None
        file_download: tuple[GeneratedFile, BinaryIO] | None = None
        preflight_headers: dict[str, str] | None = None
        try:
            parsed_path = urlsplit(self.path)
            route_path = (
                parsed_path.path if parsed_path.path == _FILES_PATH else self.path
            )
            allowed_methods = _ROUTE_METHODS.get(route_path)
            if allowed_methods is None:
                raise _HttpProblem(
                    HTTPStatus.NOT_FOUND,
                    "not_found",
                    "The requested API route does not exist.",
                )
            origin = self._allowed_origin(required=self.command == "OPTIONS")
            if self.command == "OPTIONS":
                preflight_headers = self._preflight_headers(allowed_methods)
            elif self.command not in allowed_methods:
                raise _HttpProblem(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    "method_not_allowed",
                    "The HTTP method is not supported by this API.",
                    headers={"Allow": f"{', '.join(allowed_methods)}, OPTIONS"},
                )
            elif self.command == "GET":
                if route_path == _FILES_PATH:
                    file_download = self._gaia_server.chat_service.open_file(
                        **self._read_file_request(parsed_path.query)
                    )
                else:
                    payload = {"status": "ok"}
            elif self.command == "DELETE":
                username, conversation_id = self._read_cancel_request()
                cancelled_id = self._gaia_server.chat_service.cancel(
                    conversation_id, username=username,
                )
                payload = {
                    "status": "cancelled",
                    "conversation_id": cancelled_id,
                    "message": "Stopped thinking",
                }
            elif self.path == _FEEDBACK_PATH:
                payload = self._gaia_server.chat_service.record_feedback(
                    **self._read_feedback_request()
                )
            else:
                message, username, conversation_id = self._read_chat_request()
                if self._accepts_chat_stream():
                    stream_request = (message, username, conversation_id)
                else:
                    payload = self._gaia_server.chat_service.chat(
                        message, username=username, conversation_id=conversation_id,
                    ).payload()
        except Exception as error:
            self._send_problem(
                _problem_from_error(error, feedback=self.path == _FEEDBACK_PATH),
                origin=origin,
            )
            return

        if preflight_headers is not None:
            self._send_headers(
                HTTPStatus.NO_CONTENT, origin=origin,
                content_length=0, headers=preflight_headers,
            )
        elif stream_request is not None:
            message, username, conversation_id = stream_request
            self._stream_chat(
                message, username=username,
                conversation_id=conversation_id, origin=origin,
            )
        elif file_download is not None:
            generated_file, stream = file_download
            self._send_file(generated_file, stream, origin=origin)
        else:
            self._send_json(HTTPStatus.OK, payload, origin=origin)

    do_GET = _handle_request
    do_POST = _handle_request
    do_DELETE = _handle_request
    do_OPTIONS = _handle_request
    do_PATCH = _handle_request
    do_PUT = _handle_request

    def _preflight_headers(self, allowed_methods: tuple[str, ...]) -> dict[str, str]:
        requested_method = self.headers.get("Access-Control-Request-Method")
        if (
            requested_method is not None
            and requested_method.upper() not in allowed_methods
        ):
            raise _HttpProblem(
                HTTPStatus.FORBIDDEN,
                "cors_method_forbidden",
                "The requested cross-origin method is not allowed.",
            )
        requested_headers = {
            value.strip().casefold()
            for value in self.headers.get(
                "Access-Control-Request-Headers", "",
            ).split(",")
            if value.strip()
        }
        if not requested_headers.issubset({"content-type"}):
            raise _HttpProblem(
                HTTPStatus.FORBIDDEN,
                "cors_headers_forbidden",
                "The requested cross-origin headers are not allowed.",
            )
        headers = {
            "Access-Control-Allow-Methods": f"{', '.join(allowed_methods)}, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "600",
            "Vary": _PREFLIGHT_VARY,
        }
        if self.headers.get(
            "Access-Control-Request-Private-Network", "",
        ).casefold() == "true":
            headers["Access-Control-Allow-Private-Network"] = "true"
        return headers

    @property
    def _gaia_server(self) -> GaiaHttpServer:
        return cast(GaiaHttpServer, self.server)

    def _allowed_origin(self, *, required: bool = False) -> str | None:
        origin = self.headers.get("Origin")
        if origin is None:
            if required:
                raise _HttpProblem(
                    HTTPStatus.BAD_REQUEST,
                    "cors_origin_required",
                    "A browser origin is required for this preflight request.",
                )
            return None
        if origin not in self._gaia_server.allowed_origins:
            raise _HttpProblem(
                HTTPStatus.FORBIDDEN,
                "cors_origin_forbidden",
                "This browser origin is not allowed to access Gaia.",
            )
        return origin

    def _accepts_chat_stream(self) -> bool:
        """Return whether the client requested bounded NDJSON progress events."""
        for value in self.headers.get("Accept", "").split(","):
            media_type, *parameters = value.split(";")
            if media_type.strip().casefold() != _CHAT_STREAM_MEDIA_TYPE:
                continue
            quality = 1.0
            for parameter in parameters:
                name, separator, raw_value = parameter.strip().partition("=")
                if separator and name.casefold() == "q":
                    try:
                        quality = float(raw_value)
                    except ValueError:
                        quality = 0.0
                    break
            if 0.0 < quality <= 1.0:
                return True
        return False

    def _stream_chat(
        self,
        message: str,
        *,
        username: str,
        conversation_id: str | None,
        origin: str | None,
    ) -> None:
        """Stream public progress boundaries followed by one terminal event."""
        self._send_headers(
            HTTPStatus.OK, origin=origin,
            content_type=f"{_CHAT_STREAM_MEDIA_TYPE}; charset=utf-8",
            headers={"Connection": "close"},
        )
        self.close_connection = True
        stream = _ChatStream(self._write_stream_event, conversation_id)
        try:
            self._gaia_server.chat_service.chat(
                message,
                username=username,
                conversation_id=conversation_id,
                progress_callback=stream.report_progress,
                conversation_callback=stream.deliver_conversation_id,
                response_callback=stream.deliver_response,
                cancellation=stream.cancellation,
            )
        except AgentCancelledError:
            stream.deliver_cancellation()
        except Exception as error:
            stream.deliver_error(error)

    def _write_stream_event(self, payload: Mapping[str, object]) -> bool:
        encoded = _json_bytes(payload) + b"\n"
        try:
            self.wfile.write(encoded)
            self.wfile.flush()
        except OSError:
            return False
        return True

    def _read_chat_request(self) -> tuple[str, str, str | None]:
        document = self._read_json_document()
        if document.keys() - {"message", "username", "conversation_id"}:
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "The request body contains unsupported properties.",
            )
        if "message" not in document:
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_message",
                "A message is required.",
            )
        message = document["message"]
        if not isinstance(message, str) or not message.strip():
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_message",
                "Message must be non-empty text.",
            )
        if (
            len(message) > self._gaia_server.max_message_characters
            or len(message.encode("utf-8"))
            > self._gaia_server.max_message_bytes
        ):
            raise _HttpProblem(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "message_too_large",
                "Message exceeds the configured size limit.",
            )

        username = self._read_username(document)
        conversation_id = document.get("conversation_id")
        if conversation_id is not None:
            conversation_id = self._read_conversation_id(conversation_id)
        return message, username, conversation_id

    def _read_cancel_request(self) -> tuple[str, str]:
        document = self._read_json_document()
        if set(document) != {"username", "conversation_id"}:
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Cancellation requires username and conversation_id.",
            )
        username = self._read_username(document)
        return username, self._read_conversation_id(document["conversation_id"])

    def _read_file_request(self, query: str) -> dict[str, str]:
        try:
            pairs = parse_qsl(
                query,
                keep_blank_values=True,
                strict_parsing=True,
                errors="strict",
                max_num_fields=3,
            )
        except (UnicodeError, ValueError):
            pairs = []
        document = dict(pairs)
        if (
            len(pairs) != 3
            or set(document) != {"username", "conversation_id", "file_id"}
            or re.search(r"%(?![0-9A-Fa-f]{2})", query)
        ):
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "File downloads require username, conversation_id, and file_id.",
            )
        username = self._read_username(document)
        self._read_conversation_id(document["conversation_id"])
        if not is_valid_conversation_id(document["file_id"]):
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_file_id",
                "File ID has an invalid format.",
            )
        return {**document, "username": username}

    def _read_feedback_request(self) -> dict[str, object]:
        document = self._read_json_document()
        required = {
            "username",
            "conversation_id",
            "interaction",
            "rating",
            "reasons",
            "details",
        }
        if document.keys() != required:
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Feedback requires exactly username, conversation_id, "
                "interaction, rating, reasons, and details.",
            )
        return document

    def _read_json_document(self) -> dict[str, object]:
        transfer_encoding = self.headers.get("Transfer-Encoding")
        if transfer_encoding is not None:
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Chunked request bodies are not supported.",
            )
        content_type = self.headers.get("Content-Type")
        if content_type is None or (
            content_type.split(";", 1)[0].strip().casefold()
            != "application/json"
        ):
            raise _HttpProblem(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json.",
            )
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Content-Length is required.",
            )
        try:
            body_length = int(content_length)
        except ValueError as error:
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Content-Length must be a non-negative integer.",
            ) from error
        if body_length < 0:
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Content-Length must be a non-negative integer.",
            )
        if body_length > self._gaia_server.max_request_bytes:
            self.close_connection = True
            raise _HttpProblem(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                "The JSON request body is too large.",
            )
        raw_body = self.rfile.read(body_length)
        if len(raw_body) != body_length:
            self.close_connection = True
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "The request body ended before Content-Length bytes arrived.",
            )
        try:
            body_text = raw_body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_encoding",
                "The JSON request body must be valid UTF-8.",
            ) from error
        try:
            document = json.loads(
                body_text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except ValueError as error:
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "The request body must contain valid JSON.",
            ) from error
        if not isinstance(document, dict):
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "The request body must be a JSON object.",
            )
        return document

    @staticmethod
    def _read_conversation_id(value: object) -> str:
        if not is_valid_conversation_id(value):
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_conversation_id",
                "Conversation ID has an invalid format.",
            )
        return cast(str, value)

    @staticmethod
    def _read_username(document: Mapping[str, object]) -> str:
        if "username" not in document:
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_username",
                "A username is required.",
            )
        try:
            return normalize_chat_username(document["username"])
        except (TypeError, ValueError):
            raise _HttpProblem(
                HTTPStatus.BAD_REQUEST,
                "invalid_username",
                "Username has an invalid format.",
            ) from None

    def _send_problem(self, problem: _HttpProblem, *, origin: str | None) -> None:
        self._send_json(
            problem.status,
            {"error": {"code": problem.code, "message": problem.public_message}},
            origin=origin,
            headers=problem.headers,
        )

    def _send_json(
        self,
        status: HTTPStatus,
        payload: Mapping[str, object],
        *,
        origin: str | None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        encoded = _json_bytes(payload)
        self._send_headers(
            status, origin=origin,
            content_type="application/json; charset=utf-8",
            content_length=len(encoded), headers=headers,
        )
        self.wfile.write(encoded)

    def _send_file(
        self,
        generated_file: GeneratedFile,
        stream: BinaryIO,
        *,
        origin: str | None,
    ) -> None:
        with stream:
            fallback_name = "".join(
                character
                if 32 <= ord(character) < 127 and character not in {'"', "\\"}
                else "_"
                for character in generated_file.filename
            )
            disposition = (
                f'attachment; filename="{fallback_name}"; '
                f"filename*=UTF-8''{quote(generated_file.filename, safe='')}"
            )
            try:
                self._send_headers(
                    HTTPStatus.OK,
                    origin=origin,
                    content_type=generated_file.media_type,
                    content_length=os.fstat(stream.fileno()).st_size,
                    headers={
                        "Content-Disposition": disposition,
                        "Access-Control-Expose-Headers": "Content-Disposition",
                    },
                )
                while chunk := stream.read(_FILE_CHUNK_BYTES):
                    self.wfile.write(chunk)
            except OSError:
                self.close_connection = True

    def _send_headers(
        self,
        status: HTTPStatus,
        *,
        origin: str | None,
        content_type: str | None = None,
        content_length: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        response_headers = {
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            **(headers or {}),
        }
        if content_type is not None:
            response_headers["Content-Type"] = content_type
        if content_length is not None:
            response_headers["Content-Length"] = str(content_length)
        if origin is not None:
            response_headers["Access-Control-Allow-Origin"] = origin
            response_headers.setdefault("Vary", "Origin")
        self.send_response(status.value)
        for name, value in response_headers.items():
            self.send_header(name, value)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """Keep the bridge quiet; Gaia's bounded interaction traces are canonical."""
        del format, args


def create_gaia_http_server(
    chat_service: GaiaChatService,
    *,
    host: str = DEFAULT_GAIA_HTTP_HOST,
    port: int = DEFAULT_GAIA_HTTP_PORT,
    allowed_origins: Iterable[str] = DEFAULT_CORS_ALLOWED_ORIGINS,
    max_request_bytes: int = DEFAULT_MAX_HTTP_REQUEST_BYTES,
    max_message_characters: int = DEFAULT_MAX_CHAT_MESSAGE_CHARACTERS,
    max_message_bytes: int = DEFAULT_MAX_CHAT_MESSAGE_BYTES,
) -> GaiaHttpServer:
    """Bind the threaded HTTP server; the caller starts serving requests."""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("HTTP host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("HTTP port must be an integer between 0 and 65535")
    return GaiaHttpServer(
        (host, port),
        chat_service,
        allowed_origins=allowed_origins,
        max_request_bytes=max_request_bytes,
        max_message_characters=max_message_characters,
        max_message_bytes=max_message_bytes,
    )


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON property: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"unsupported JSON constant: {value}")
