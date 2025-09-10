"""Job event models for cost and usage tracking."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum, auto
from uuid import UUID

from pydantic import BaseModel, Field, JsonValue, field_validator


class ExecutionType(StrEnum):
    """Type of execution for job events."""

    TRAJECTORY = auto()
    SESSION = auto()


class CostComponent(StrEnum):
    """Cost component types for job events."""

    LLM_USAGE = auto()
    EXTERNAL_SERVICE = auto()
    STEP = auto()


class JobEventCreateRequest(BaseModel):
    """Request model for creating a job event matching crow-service schema."""

    execution_id: UUID = Field(description="UUID for trajectory_id or session_id")
    execution_type: ExecutionType = Field()
    cost_component: CostComponent = Field()
    started_at: datetime = Field(description="Start time of the cost period")
    ended_at: datetime = Field(description="End time of the cost period")
    crow: str | None = Field(
        default=None, description="Foreign key to the crows db key name"
    )
    amount_acu: float | None = Field(default=None, description="Cost amount in ACUs")
    amount_usd: float | None = Field(default=None, description="Cost amount in USD")
    rate: float | None = Field(default=None, description="Rate per token/call in USD")
    input_token_count: int | None = Field(
        default=None, description="Input token count for LLM calls"
    )
    completion_token_count: int | None = Field(
        default=None, description="Completion token count for LLM calls"
    )
    metadata: dict[str, JsonValue] | None = Field(
        default=None, description="Additional metadata"
    )

    @field_validator("amount_acu", "amount_usd", "rate", mode="before")
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert float values to Decimal for database storage."""
        if v is not None:
            return Decimal(str(v))
        return v


class JobEventUpdateRequest(BaseModel):
    """Request model for updating a job event matching crow-service schema."""

    amount_acu: float | None = Field(default=None, description="Cost amount in ACUs")
    amount_usd: float | None = Field(default=None, description="Cost amount in USD")
    rate: float | None = Field(default=None, description="Rate per token/call in USD")
    input_token_count: int | None = Field(
        default=None, description="Input token count for LLM calls"
    )
    completion_token_count: int | None = Field(
        default=None, description="Completion token count for LLM calls"
    )
    metadata: dict[str, JsonValue] | None = Field(
        default=None, description="Additional metadata"
    )
    started_at: datetime | None = Field(
        default=None, description="Start time of the cost period"
    )
    ended_at: datetime | None = Field(
        default=None, description="End time of the cost period"
    )

    @field_validator("amount_acu", "amount_usd", "rate", mode="before")
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert float values to Decimal for database storage."""
        if v is not None:
            return Decimal(str(v))
        return v


class JobEventCreateResponse(BaseModel):
    """Response model for job event creation."""

    id: UUID = Field(description="UUID of the created job event")
