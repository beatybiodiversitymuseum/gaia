"""Run runtime coroutine tests on the application's asyncio backend."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
