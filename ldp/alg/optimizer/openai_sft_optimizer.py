"""This module defines an expert iteration optimizer for black-box OpenAI LLMs.

The optimizer manages the collation and formatting of training rollout data and initiates fine-tuning jobs through
OpenAI's API:

https://platform.openai.com/docs/guides/fine-tuning/analyzing-your-fine-tuned-model

For expert iteration see:

Havrilla et al. 2024. Teaching large language models to reason with reinforcement learning.
arXiv preprint arXiv:2403.04642. (https://arxiv.org/pdf/2403.04642)

Example Usage:
    - Instantiate the `BlackBoxLLMSFTOpt` with the necessary configuration.
    - Accumulate training rollout examples by calling `aggregate_trajectory`.
    - Update the model by invoking `update`, which prepares the training data,
      uploads it, and triggers the fine-tuning process.
"""

import json
import logging
import tempfile
import time
from collections.abc import Callable
from typing import Any, Self, cast

import openai
from pydantic import BaseModel, ConfigDict, Field

from ldp.agent import ReActAgent
from ldp.alg.optimizer.opt import Optimizer
from ldp.data_structures import Trajectory
from ldp.graph.common_ops import LLMCallOp
from ldp.graph.ops import OpResult

logger = logging.getLogger(__name__)


class OpenAISFTOptConfig(BaseModel):
    """Configuration class for the BlackBoxLLMSFTOpt optimizer.

    This class holds various configuration parameters for the optimizer.
    """

    lr: float = 0.001
    num_epochs: int = 1
    buffer_size: int | None = Field(
        default=None,
        description="Maximum number of finetuning examples to accumulate. "
        "If None, the buffer has no size limit.",
    )
    val_frac: float = 0.1
    reward_discount: float = 1.0  # Discount factor in [0, 1] for rewards
    return_threshold: float | None = Field(
        default=None,
        description="Minimum return required for a trajectory to be added to the training buffer. If None, "
        "all trajectories are added.",
    )


class OpenAISFTOpt(BaseModel, Optimizer):
    """An optimizer for finetuning black-box LLMs that interact via an API.

    Expert Iteration (SFT) optimizer for fine-tuning black-box OpenAI LLMs.
    It handles the aggregation of training data, manages a buffer of training examples,
    and initiates fine-tuning jobs via the API.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config: OpenAISFTOptConfig = Field(default_factory=OpenAISFTOptConfig)
    log_to_wandb: bool = False
    llm_call_op: LLMCallOp
    client: openai.OpenAI = Field(default_factory=openai.OpenAI)

    # State
    train_buffer: list = Field(default_factory=list)
    val_buffer: list = Field(default_factory=list)
    fine_tune_job_id: str | None = None

    train_dataset: list[Any] | None = None
    val_dataset: list[Any] | None = None

    def __init__(self, **data):
        super().__init__(**data)

        # Validate and populate the training and validation buffers
        if self.train_dataset:
            self.train_buffer.extend(self.train_dataset)
        if self.val_dataset:
            self.val_buffer.extend(self.val_dataset)

    @property
    def buffer_is_full(self) -> bool:
        return (
            self.config.buffer_size is not None
            and len(self.train_buffer) >= self.config.buffer_size
        )

    def aggregate_trajectory(
        self,
        trajectory: Trajectory,
        buffer_type: str = "train",
        len_penalty_fn: Callable[[int], float] | None = None,
    ) -> None:
        """Adds training rollout examples from a trajectory to the training buffer.

        This method extracts rollouts and their corresponding discounted returns from a trajectory and stores them
        in the appropriate buffer (training or validation) if they meet the return threshold criteria.

        We apply a weight of 1 to actions and a weight of 0 to states. This reflects the fact that we want to train the
        agent using P(action | state) as the target distribution. Note that in the OpenAI API, the weight may only be
        applied to assistant messages.

        Args:
            trajectory: The trajectory containing rollouts and rewards.
            buffer_type: The buffer to which the trajectory should be added. Must be either "train" or "validation".
            len_penalty_fn: An optional callable that takes an integer (the length of
                the list of discounted returns) and returns a scalar penalty to be applied to the discounted return.

        Raises:
            RuntimeError: If a rollout in the trajectory does not have an associated compute graph.
            ValueError: If the supplied buffer type is invalid. Must be either "train" or "validation".
        """
        # Validate buffer type
        if buffer_type not in {"train", "validation"}:
            raise ValueError('buffer_type must be either "train" or "validation".')

        # Compute the discounted returns
        discounted_returns = trajectory.compute_discounted_returns(
            self.config.reward_discount
        )

        # Apply the penalty on the length of the trajectory if a penalty function is provided
        if len_penalty_fn is not None:
            penalty = len_penalty_fn(len(discounted_returns))
            modified_return = discounted_returns[0] * penalty
        else:
            modified_return = discounted_returns[0]

        # Don't add trajectory to the buffer if it failed or doesn't meet the return threshold
        if trajectory.failed or (
            self.config.return_threshold is not None
            and modified_return < self.config.return_threshold
        ):
            return

        traj_msgs = []
        for step in trajectory.steps:
            action_call_id = cast(OpResult, step.action).call_id
            if action_call_id is None:
                raise RuntimeError("Received an action without compute graph attached.")
            call_ids = self.llm_call_op.get_call_ids({action_call_id.run_id})
            for call_id in call_ids:
                if (
                    self.llm_call_op.ctx.get(call_id, "grad_output", default=None)
                    is None
                ):
                    # This op call was pruned from backward compute graph - skip.
                    continue

                _, input_kwargs = self.llm_call_op.ctx.get(call_id, "input")
                outputs = self.llm_call_op.ctx.get(call_id, "output").value.model_dump()

                # Add "weight": 1 to the outputs dictionary. NB: weight should ONLY be added to assistant messages. All
                # output messages are assumed to be assistant messages and will throw an error otherwise.
                outputs["weight"] = 1

                # Just supply list of messages here. Call model_dump on each element of list. Add weight = 0 for input
                traj_msgs += [
                    {
                        **msg.model_dump(),
                        **({"weight": 0} if msg.role == "assistant" else {}),
                    }
                    for msg in OpResult.unwrap_value(input_kwargs["msgs"])
                ]
                traj_msgs.append(outputs)

        # Choose the appropriate buffer
        target_buffer = self.train_buffer if buffer_type == "train" else self.val_buffer

        # Add trajectory to the specified buffer. Buffer is List[List[dict]]
        target_buffer.append(traj_msgs)

        # If buffer size is set, ensure that the buffer does not exceed the specified size. If it does exceed the size
        # remove the oldest samples.
        if (
            self.config.buffer_size is not None
            and len(target_buffer) >= self.config.buffer_size
        ):
            # Calculate the starting index for slicing
            start_index = len(target_buffer) - self.config.buffer_size
            # Assign the last `buffer_size` elements to `target_buffer`
            target_buffer[:] = target_buffer[start_index:]

    async def update(self, check_for_completion: bool = False):
        """Updates the model parameters based on the accumulated training data.

        This method processes the accumulated training data by formatting it into the appropriate structure for the
        API, uploads it, and then initiates a fine-tuning job. It is important to note that the OpenAI finetuning API
        has a minimum requirement of 10 training examples (trajectories) to perform fine-tuning.

        Args:
            check_for_completion: A flag to indicate whether to check for the completion of the fine-tuning job.

        Raises:
            ValueError: If the training data fails to upload or the fine-tuning job fails to start.
        """
        # Prepare the data for fine-tuning in chat format
        training_data = [{"messages": traj} for traj in self.train_buffer]
        validation_data = (
            [{"messages": traj} for traj in self.val_buffer]
            if self.val_buffer
            else None
        )

        if not training_data:
            return

        def write_to_tempfile(data):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".jsonl"
            ) as temp_file:
                for example in data:
                    temp_file.write((json.dumps(example) + "\n").encode("utf-8"))
                return temp_file.name

        train_temp_file_path = write_to_tempfile(training_data)
        val_temp_file_path = (
            write_to_tempfile(validation_data) if validation_data else None
        )

        try:
            with open(train_temp_file_path, "rb") as train_file:
                file_id = self.client.files.create(
                    file=train_file, purpose="fine-tune"
                ).id

            val_file_id = None
            if val_temp_file_path:
                with open(val_temp_file_path, "rb") as val_file:
                    val_file_id = self.client.files.create(
                        file=val_file, purpose="fine-tune"
                    ).id

            fine_tune_job = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                validation_file=val_file_id,
                model="gpt-3.5-turbo",
            )

            self.fine_tune_job_id = fine_tune_job.id
            logger.info(f"Fine-tuning job created with ID: {self.fine_tune_job_id}")

            # Check the status of the job periodically until it completes
            if check_for_completion:
                while True:
                    job_status = self.client.fine_tuning.jobs.retrieve(
                        self.fine_tune_job_id
                    )
                    status = job_status.status

                    if status == "succeeded":
                        logger.info("Fine-tuning job succeeded.")
                        break
                    if status == "failed":
                        logger.error(
                            f"Fine-tuning job failed with status: {job_status}"
                        )
                        raise ValueError(
                            f"Fine-tuning job failed with status: {job_status}"
                        )
                    logger.info(
                        f"Fine-tuning job is still running. Current status: {status}"
                    )
                    time.sleep(30)  # Wait 30 seconds before checking the status again

        except (openai.APIConnectionError, openai.RateLimitError, openai.APIError) as e:
            logger.exception("Error during fine-tuning job creation")
            raise ValueError("Failed to create the fine-tuning job.") from e

    def clear_train_buffer(self):
        """Clear the training buffer."""
        self.train_buffer.clear()

    def clear_val_buffer(self):
        """Clear the validation buffer."""
        self.val_buffer.clear()

    @classmethod
    def from_agent(cls, agent: ReActAgent, **kwargs) -> Self:
        """Creates an instance of the OpenAISFTOpt class from an existing ReActAgent by extracting.

        the LLM call operation (llm_call_op) from the provided ReActAgent. At the moment, only initialization
        from ReActAgent is supported.

        Args:
            agent: The ReActAgent from which to extract the LLM call operation.
            **kwargs: Additional keyword arguments to pass to the OpenAISFTOpt constructor.

        Returns:
            OpenAISFTOpt: An instance of the OpenAISFTOpt class initialized with the LLM call
            operation from the provided ReActAgent.
        """
        return cls(
            llm_call_op=agent._react_module.tool_select_module.llm_call_op,
            **kwargs,
        )
