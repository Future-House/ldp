from typing import TypeVar

import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput

TOutputType = TypeVar("TOutputType", torch.Tensor, GenerateDecoderOnlyOutput)


class TensorChunker:
    """Splits tensors into chunks and adds dummy chunks as needed for parallel processing frameworks like FSDP."""

    def __init__(self, num_chunks: int):
        self.num_chunks = num_chunks

    def chunkify(self, *args, **kwargs) -> tuple[list[tuple], list[dict], list[bool]]:
        """Splits the args into self.num_chunks chunks, adding dummy chunks as needed.

        Returns:
            tuple: Contains:
                - A list of len self.num_chunks, where each element is a tuple of split args.
                - A list of len self.num_chunks, where each element is a dict of split kwargs.
                - A list of len self.num_chunks, where each element is a bool indicating
                  whether the chunk is a dummy chunk.

        NOTE: the below Note is not always true - should come back and fix.
            Can be seen by observing e.g. `len(torch.chunk(torch.arange(20), 8, dim=0)) == 7`.
            Not necessarily a huge problem - just means we'll be chunking suboptimally.
            But there may be cases where that causes problems.
        # Note: Dummy flags will be set only if the incoming batch is smaller than self.num_chunks,
        # resulting in empty chunks insertion.

        Example:
        ```python
        input_tensor = torch.randn(10, 5)  # A tensor with batch size 10
        some_int = 42  # An integer value that isn't split

        chunker = TensorChunker(num_chunks=4)
        split_args, split_kwargs, dummy_flags = chunker.chunkify(
            input_tensor=input_tensor, some_int=some_int
        )
        ```

        Output split_args:
        [
            ((input_tensor_chunk_0, ), (some_int, )),  # batch of 3
            ((input_tensor_chunk_1, ), (some_int, )),  # batch of 3
            ((input_tensor_chunk_2, ), (some_int, )),  # batch of 3
            ((input_tensor_chunk_3, ), (some_int, )),  # batch of 1
        ],
        dummy_flags = [False, False, False, False]

        """
        # Initialize lists to hold split args and kwargs
        split_args_list: list[list] = [[] for _ in range(self.num_chunks)]
        split_kwargs_list: list[dict] = [{} for _ in range(self.num_chunks)]
        dummy_chunk_flags: list[bool] = []

        def process_items(items, is_kwargs=False):
            """Helper function to process both args and kwargs."""
            nonlocal dummy_chunk_flags
            for item in items:
                key, value = item if is_kwargs else (None, item)
                chunks, out_chunk_dummy_flags = self._split_value(value)
                for i in range(self.num_chunks):
                    if is_kwargs:
                        split_kwargs_list[i][key] = chunks[i]
                    else:
                        split_args_list[i].append(chunks[i])

                # Ensure dummy chunk flags are consistent across all items
                if dummy_chunk_flags and out_chunk_dummy_flags:
                    assert dummy_chunk_flags == out_chunk_dummy_flags, (
                        "All inputs must result in the same dummy chunk flags."
                    )
                dummy_chunk_flags = out_chunk_dummy_flags or dummy_chunk_flags

        # Process args and kwargs
        process_items(args)
        process_items(kwargs.items(), is_kwargs=True)

        split_args_tuples = [tuple(args) for args in split_args_list]
        return split_args_tuples, split_kwargs_list, dummy_chunk_flags

    @staticmethod
    def dechunkify(
        outputs_list: list[TOutputType],
        dummy_chunk_flags: list[bool],
    ) -> TOutputType:
        """Reassembles the outputs from the handlers, removing outputs corresponding to dummy chunks.

        The desired behavior is such that the following are equivalent:
        ```python
        # "single device"
        output_single = module(input)

        # "multi device" (in reality each device would only run one chunk)
        chunks, dummy_chunk_flags = chunker.chunkify(input)
        output_multi = chunker.dechunkify(
            [module(chunk) for chunk in chunks], dummy_chunk_flags
        )
        ```
        """
        # Filter out outputs corresponding to dummy chunks
        real_outputs = [
            output
            for output, is_dummy in zip(outputs_list, dummy_chunk_flags, strict=True)
            if not is_dummy
        ]

        if isinstance(real_outputs[0], torch.Tensor):
            # If we received a list of tensors, concat along the batch dimension
            # TODO: consider how/if to handle differently shaped tensors
            return torch.cat(real_outputs, dim=0)

        if isinstance(real_outputs[0], GenerateDecoderOnlyOutput):
            sequences: list[torch.Tensor] = []
            for output in real_outputs:
                sequences.extend(output.sequences)
            scores: list[list[torch.Tensor | None]] | None = None
            if real_outputs[0].scores is not None:
                assert all(output.scores is not None for output in real_outputs)
                # `output.scores` is a tuple (one el per decoded token), with each tensor of shape
                # (batch_size, vocab_size).
                # To get the scores for the `i`-th batch element, we'd do `[score[i] for score in output.scores]`.
                # `output.scores` will have different lengths across workers, so we cannot simply concatenate them.
                # Adding dummy scores is error-prone and loses info. Instead, we create a merged `scores` with the
                # same access semantics. `score[i]` will be a tensor if it exists for batch element `i` and `None`
                # otherwise.
                scores = []
                max_output_len = max(
                    len(output.scores or []) for output in real_outputs
                )
                for i in range(max_output_len):
                    scores_step_i: list[torch.Tensor | None] = []
                    for output in real_outputs:
                        if output.scores is not None and i < len(output.scores):
                            scores_step_i.extend(list(output.scores[i]))
                        else:
                            # Add bsz Nones for this worker, since it did not reach token `i`.
                            scores_step_i.extend([None] * output.sequences.shape[0])

                    scores.append(scores_step_i)

            return GenerateDecoderOnlyOutput(
                sequences=sequences,  # type: ignore[arg-type]
                scores=scores,  # type: ignore[arg-type]
            )

        raise ValueError("Unsupported output type")

    def _split_value(self, value):
        """
        Splits a single value into chunks, adding a dummy value if necessary.

        Right now, only torch.Tensor values are split. Non-tensor values are replicated.
        """
        if isinstance(value, torch.Tensor):
            chunks = list(torch.chunk(value, self.num_chunks, dim=0))
            dummy_chunk_flags = []
            for i in range(self.num_chunks):
                if i >= len(chunks):
                    # Chunk 0 will always exist, and we need only a batch of one ([:1])
                    # to activate the model.
                    # We use the first element of the existing chunks as real data to avoid
                    # errors in the model that may expect a specific token structure.
                    chunks.append(chunks[0][:1])
                    dummy_chunk_flags.append(True)
                else:
                    dummy_chunk_flags.append(False)

            return chunks, dummy_chunk_flags
        # Non-tensor values are replicated
        return [value] * self.num_chunks, None
