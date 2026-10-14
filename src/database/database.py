"""Build a PostgreSQL SQLAlchemy engine from local environment settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import URL, create_engine, make_url
from sqlalchemy.engine import Engine


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPOSITORY_ROOT / ".env"


class DatabaseConfigurationError(ValueError):
    """Indicate that database connection settings are missing or invalid."""


def _setting(
    name: str,
    *,
    environment: Mapping[str, str],
    env_file_values: Mapping[str, str | None],
    default: str | None = None,
) -> str:
    """Return one environment-first, non-blank database setting."""
    value = environment.get(name)
    if value is None:
        value = env_file_values.get(name)
    if value is None:
        value = default
    if value is None or value == "":
        raise DatabaseConfigurationError(f"missing database setting: {name}")
    return value


def load_database_url(
    *,
    environment: Mapping[str, str] | None = None,
    env_path: Path = DEFAULT_ENV_PATH,
) -> URL:
    """Load a psycopg SQLAlchemy URL without exposing its password in text."""
    active_environment = os.environ if environment is None else environment
    env_file_values = (
        dotenv_values(env_path, encoding="utf-8") if env_path.is_file() else {}
    )

    configured_url = active_environment.get("DATABASE_URL")
    if configured_url:
        url = make_url(configured_url)
        if url.get_backend_name() != "postgresql":
            raise DatabaseConfigurationError(
                "DATABASE_URL must use a PostgreSQL dialect"
            )
        return url

    port_text = _setting(
        "POSTGRES_PORT",
        environment=active_environment,
        env_file_values=env_file_values,
        default="5432",
    )
    try:
        port = int(port_text)
    except ValueError as error:
        raise DatabaseConfigurationError(
            f"POSTGRES_PORT must be an integer: {port_text}"
        ) from error
    if not 1 <= port <= 65535:
        raise DatabaseConfigurationError(
            f"POSTGRES_PORT must be between 1 and 65535: {port}"
        )

    return URL.create(
        "postgresql+psycopg",
        username=_setting(
            "POSTGRES_USER",
            environment=active_environment,
            env_file_values=env_file_values,
        ),
        password=_setting(
            "POSTGRES_PASSWORD",
            environment=active_environment,
            env_file_values=env_file_values,
        ),
        host=_setting(
            "POSTGRES_HOST",
            environment=active_environment,
            env_file_values=env_file_values,
            default="127.0.0.1",
        ),
        port=port,
        database=_setting(
            "POSTGRES_DB",
            environment=active_environment,
            env_file_values=env_file_values,
        ),
    )


def create_database_engine(
    *,
    environment: Mapping[str, str] | None = None,
    env_path: Path = DEFAULT_ENV_PATH,
) -> Engine:
    """Create a synchronous SQLAlchemy engine for Gaia's PostgreSQL database."""
    return create_engine(
        load_database_url(environment=environment, env_path=env_path),
        pool_pre_ping=True,
    )
