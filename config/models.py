"""Pydantic configuration models for Fara E2E agent."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator


# Load .env file if present
load_dotenv()


class AgentConfig(BaseModel):
    """LLM agent configuration."""

    model: str = Field(
        default="microsoft_fara-7b",
        description="Model name to use for the LLM",
    )
    base_url: str = Field(
        default="http://localhost:1234/v1",
        description="Base URL for the LLM API endpoint",
    )
    api_key: str = Field(
        default="lm-studio",
        description="API key for the LLM service",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    max_rounds: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of action rounds per test",
    )
    max_tokens: int = Field(
        default=768,
        ge=100,
        le=4096,
        description="Maximum tokens for model response",
    )
    max_n_images: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of images to include in context",
    )
    debug_log_requests: bool = Field(
        default=False,
        description="Log full LLM request payloads (for debugging context/loops)",
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure base_url doesn't have trailing slash."""
        return v.rstrip("/")

    @model_validator(mode="before")
    @classmethod
    def load_from_env(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Load values from environment variables if not explicitly set."""
        env_mapping = {
            "base_url": "FARA_BASE_URL",
            "api_key": "FARA_API_KEY",
            "model": "FARA_MODEL",
        }
        for field_name, env_var in env_mapping.items():
            if field_name not in data or data[field_name] is None:
                env_value = os.getenv(env_var)
                if env_value:
                    data[field_name] = env_value
        return data


class BrowserConfig(BaseModel):
    """Browser automation configuration."""

    browser: Literal["chromium", "firefox", "webkit"] = Field(
        default="firefox",
        description="Browser engine to use",
    )
    headless: bool = Field(
        default=True,
        description="Run browser in headless mode",
    )
    viewport_width: int = Field(
        default=1440,
        ge=800,
        le=3840,
        description="Browser viewport width",
    )
    viewport_height: int = Field(
        default=900,
        ge=600,
        le=2160,
        description="Browser viewport height",
    )
    show_overlay: bool = Field(
        default=False,
        description="Show debug overlay in browser",
    )
    show_click_markers: bool = Field(
        default=False,
        description="Show click markers during automation",
    )
    slow_mo: int = Field(
        default=0,
        ge=0,
        le=5000,
        description="Slow down browser operations by this many ms",
    )

    @model_validator(mode="after")
    def set_overlay_defaults(self) -> "BrowserConfig":
        """Enable overlays by default in headful mode."""
        if not self.headless:
            # Only set defaults if not explicitly configured
            object.__setattr__(self, "show_overlay", True)
            object.__setattr__(self, "show_click_markers", True)
        return self


class ReportingConfig(BaseModel):
    """Reporting and output configuration."""

    save_screenshots: bool = Field(
        default=True,
        description="Save screenshots during test execution",
    )
    screenshots_folder: Path = Field(
        default=Path("./screenshots"),
        description="Directory for saving screenshots",
    )
    reports_folder: Path = Field(
        default=Path("./reports"),
        description="Directory for saving reports",
    )
    downloads_folder: Path = Field(
        default=Path("./downloads"),
        description="Directory for downloaded files",
    )
    output_format: Literal["html", "json", "junit", "all"] = Field(
        default="html",
        description="Report output format",
    )
    embed_screenshots: bool = Field(
        default=False,
        description="Embed screenshots as base64 in HTML reports",
    )

    @field_validator("screenshots_folder", "reports_folder", "downloads_folder", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v)
        return v


class FaraConfig(BaseModel):
    """Root configuration model combining all config sections."""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    # Execution settings
    parallel_workers: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of parallel test workers",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (pause on each step)",
    )

    @classmethod
    def from_flat_dict(cls, data: dict[str, Any]) -> "FaraConfig":
        """Create config from a flat dictionary (legacy format compatibility)."""
        # Map flat keys to nested structure
        agent_keys = {
            "model", "base_url", "api_key", "temperature",
            "max_rounds", "max_tokens", "max_n_images", "debug_log_requests"
        }
        browser_keys = {
            "browser", "headless", "viewport_width", "viewport_height",
            "show_overlay", "show_click_markers", "slow_mo"
        }
        reporting_keys = {
            "save_screenshots", "screenshots_folder", "reports_folder",
            "downloads_folder", "output_format", "embed_screenshots"
        }

        nested = {
            "agent": {},
            "browser": {},
            "reporting": {},
        }

        for key, value in data.items():
            if key in agent_keys:
                nested["agent"][key] = value
            elif key in browser_keys:
                nested["browser"][key] = value
            elif key in reporting_keys:
                nested["reporting"][key] = value
            elif key == "parallel_workers":
                nested["parallel_workers"] = value
            elif key == "verbose":
                nested["verbose"] = value
            elif key == "debug":
                nested["debug"] = value
            # Legacy mappings
            elif key == "screenshots_folder":
                nested["reporting"]["screenshots_folder"] = value
            elif key == "downloads_folder":
                nested["reporting"]["downloads_folder"] = value

        return cls.model_validate(nested)


def load_config(
    config_path: Optional[Path] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> FaraConfig:
    """
    Load configuration from file with CLI overrides.

    Priority (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. Config file
    4. Defaults
    """
    config_data: dict[str, Any] = {}

    # Load from file if provided or default exists
    if config_path is None:
        config_path = Path("config.json")

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix in {".yaml", ".yml"}:
                import yaml
                config_data = yaml.safe_load(f) or {}
            else:
                config_data = json.load(f)

    # Check if it's flat or nested format
    is_flat = any(key in config_data for key in ["model", "base_url", "api_key"])

    if is_flat:
        config = FaraConfig.from_flat_dict(config_data)
    else:
        config = FaraConfig.model_validate(config_data)

    # Apply CLI overrides
    if cli_overrides:
        config_dict = config.model_dump()
        _apply_overrides(config_dict, cli_overrides)
        config = FaraConfig.model_validate(config_dict)

    return config


def _apply_overrides(config_dict: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply CLI overrides to config dictionary."""
    override_mapping = {
        "browser": ("browser", "browser"),
        "headless": ("browser", "headless"),
        "headful": ("browser", "headless"),  # inverted
        "parallel": ("parallel_workers", None),
        "verbose": ("verbose", None),
        "debug": ("debug", None),
        "output_format": ("reporting", "output_format"),
        "base_url": ("agent", "base_url"),
    }

    for key, value in overrides.items():
        if value is None:
            continue

        if key == "headful":
            config_dict["browser"]["headless"] = not value
            continue

        mapping = override_mapping.get(key)
        if mapping:
            section, field = mapping
            if field is None:
                config_dict[section] = value
            else:
                config_dict[section][field] = value

