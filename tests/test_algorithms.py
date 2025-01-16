import operator

import pytest
from aviary.core import DummyEnv
from aviary.utils import MultipleChoiceQuestion

from ldp.agent import SimpleAgent
from ldp.alg import bulk_evaluate_consensus, compute_pass_at_k
from ldp.utils import discounted_returns


@pytest.mark.asyncio
async def test_rollout_and_discounting(dummy_env: DummyEnv) -> None:
    obs, tools = await dummy_env.reset()

    agent = SimpleAgent(tools=tools)
    agent_state = await agent.init_state(tools=tools)

    observations = []
    actions = []
    rewards = []
    terms = []
    done = True
    for i in range(3):  # noqa: B007
        if done:
            obs, _ = await dummy_env.reset()
            agent_state = await agent.init_state(tools=tools)

        observations.append((obs, agent_state))
        action, agent_state, _ = await agent.get_asv(agent_state, obs)
        obs, reward, done, _ = await dummy_env.step(action.value)
        actions.append(action)
        rewards.append(reward)
        terms.append(done)

    print(terms)
    d_returns = discounted_returns(rewards, terms, 0.5)
    print(d_returns)


@pytest.mark.asyncio
async def test_consensus_evaluation() -> None:
    # We have two questions, so let's group based on question
    question_1 = MultipleChoiceQuestion(
        question="What is the meaning of life?",
        options=["-84", "11", "cheesecake"],
        ideal_answer="42",
    )
    question_2 = MultipleChoiceQuestion(
        question="What is a healthy fruit?",
        options=["brownie", "chocolate bar", "french fry"],
        ideal_answer="apple",
    )
    question_3 = MultipleChoiceQuestion(
        question="What is the highest number?",
        options=["1", "2", "4"],
        ideal_answer="8",
    )
    data_with_several_groups: list[tuple[MultipleChoiceQuestion, str]] = [
        # Has consensus and it was correct
        (question_1, "-84"),
        (question_1, "11"),
        (question_1, "11"),
        (question_1, "cheesecake"),
        (question_1, "42"),
        (question_1, "42"),
        (question_1, "42"),
        (question_1, "42"),
        (question_1, "42"),
        (question_1, "42"),
        # Has consensus and it was correct
        (question_2, "brownie"),
        (question_2, "brownie"),
        (question_2, "apple"),
        (question_2, "apple"),
        (question_2, "apple"),
        (question_2, "apple"),
        (question_2, "apple"),
        (question_2, "apple"),
        # Has no consensus and regardless it's incorrect
        (question_3, "1"),
        (question_3, "2"),
        (question_3, "1"),
        (question_3, "2"),
    ]
    # NOTE: this consensus is sensitive to seed
    expected_consensus = {
        question_1.question: [("42", 3), ("11", 1), ("-84", 1)],
        question_2.question: [("apple", 4), ("brownie", 1)],
        question_3.question: [("1", 3), ("2", 2)],
    }

    # Check accuracy is 0% without an ideal answer
    assert await bulk_evaluate_consensus(
        data_with_several_groups,
        grouping_fn=lambda x: x[0].question,
        extract_answer_fn=operator.itemgetter(1),
        num_samples=5,
        seed=42,
    ) == (expected_consensus, 0.0)
    # Check accuracy is present when we can get an ideal answer
    assert await bulk_evaluate_consensus(
        data_with_several_groups,
        grouping_fn=lambda x: x[0].question,
        extract_answer_fn=operator.itemgetter(1),
        ideal_answer_fn=lambda x: x[0].ideal_answer,
        num_samples=5,
        seed=42,
    ) == (expected_consensus, 2 / 3)


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        pytest.param(10, 10, 3, 1.0, id="all-correct-k>1"),
        pytest.param(10, 10, 1, 1.0, id="all-correct-k=1"),
        pytest.param(10, 0, 3, 0.0, id="all-incorrect-k>1"),
        pytest.param(10, 0, 1, 0.0, id="all-incorrect-k=1"),
        pytest.param(3, 1, 3, 1.0, id="n-c<k"),
        (10, 5, 3, 1 - 1 / 12),
        (2, 1, 1, 1 / 2),  # Match https://ai.stackexchange.com/a/40396
        # SEE: https://github.com/parker-research/pass-at-k/blob/037c5d477486f9e95e1c21fc349576447cd6ce8b/tests/test_pass_at_k.py
    ],
)
def test_compute_pass_at_k(n: int, c: int, k: int, expected: float) -> None:
    assert compute_pass_at_k(n, c, k) == pytest.approx(expected)
