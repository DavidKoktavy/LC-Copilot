import asyncio
import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="lc-copilot-tests-"))
os.environ.update({
    "LC_LLM_PROVIDER": "offline",
    "LC_TRACE_DIR": str(_TMP / "runs"),
    "LC_DB_PATH": str(_TMP / "portfolio.db"),
})

from lc_copilot.config import get_settings
from lc_copilot.services import build_services

get_settings.cache_clear()


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture
def svc(settings):
    return run(build_services(settings))
