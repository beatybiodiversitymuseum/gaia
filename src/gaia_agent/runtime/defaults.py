"""Dependency-free defaults shared by Gaia's command-line and web entry points."""

DEFAULT_GAIA_HTTP_HOST = "127.0.0.1"
DEFAULT_GAIA_HTTP_PORT = 8765
MAX_TOOL_CALLS_PER_TURN = 15
CONTEXT_COMPACTED_NOTICE = (
    "Earlier conversation has been compacted. Continuing with its summary."
)
CONTEXT_CONTINUATION_PROMPT = (
    "Continue the pending request using the cumulative conversation summary that follows."
)
DEFAULT_MAX_ACTIVE_CHATS = 8
DEFAULT_MAX_WAITING_CHATS = 16
DEFAULT_MAX_ACTIVE_QUERIES = 8
DEFAULT_MAX_WAITING_QUERIES = 32
DEFAULT_MAX_ACTIVE_EXPORTS = 2
DEFAULT_MAX_WAITING_EXPORTS = 8
