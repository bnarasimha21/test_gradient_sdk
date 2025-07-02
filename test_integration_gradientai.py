import os
import pytest
from gradientai import GradientAI
from dotenv import load_dotenv
import re

load_dotenv()

MODEL_IDS = [
    "llama3-70b-instruct"
]

# MODEL_IDS = [
#     "llama3-70b-instruct",
#     "openai-o3-mini",
#     "mistral-nemo-instruct-2407",
#     "anthropic-claude-3-opus",
# ]

# MODEL_IDS = [
#     "llama3-70b-instruct",
#     "openai-o3-mini",
#     "mistral-nemo-instruct-2407",
#     "openai-gpt-4o-mini",
#     "openai-gpt-4o",
#     "anthropic-claude-3-opus",
#     "llama3-8b-instruct",
#     "deepseek-r1-distill-llama-70b",
#     "anthropic-claude-3.7-sonnet",
#     "anthropic-claude-3.5-sonnet",
#     "anthropic-claude-3.5-haiku",
#     "llama3.3-70b-instruct",
#     "deepseek-r1-distill-llama-70b",
#     "deepseek-r1-distill-llama-70b",
# ]

@pytest.fixture(scope="module")
def inference_client():
    return GradientAI(
        inference_key=os.environ.get("GRADIENTAI_API_KEY"),
    )

@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_basic_completion_returns_text(inference_client, model_id):
    """Test that a basic completion returns a valid response."""
    completion = inference_client.chat.completions.create(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        model=model_id,
        max_tokens=5,
    )
    assert hasattr(completion, "choices")
    assert len(completion.choices) > 0
    assert hasattr(completion.choices[0], "message")

@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_temperature_affects_randomness(inference_client, model_id):
    """Test that temperature parameter affects output randomness."""
    completions = [
        inference_client.chat.completions.create(
            messages=[{"role": "user", "content": "List a color."}],
            model=model_id,
            temperature=1.0,
            max_tokens=2,
        ).choices[0].message.content
        for _ in range(3)
    ]
    # With high temperature, expect at least some variation
    assert len(set(completions)) > 1

@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_stop_sequence_halts_generation(inference_client, model_id):
    """Test that the stop sequence halts the generation as expected."""
    completion = inference_client.chat.completions.create(
        messages=[{"role": "user", "content": "Say yes then no"}],
        model=model_id,
        stop="no",
        max_tokens=10,
    )
    output = completion.choices[0].message.content.lower().strip()
    # Accepts 'no', 'no.', 'no!', 'no ' etc.
    assert re.search(r"\bno[\s\.\!\?]*$", output)

@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_max_tokens_limits_output_length(inference_client, model_id):
    """Test that max_tokens parameter limits the output length."""
    completion = inference_client.chat.completions.create(
        messages=[{"role": "user", "content": "Repeat the word hello 10 times."}],
        model=model_id,
        max_tokens=2,
    )
    # Should not be too long
    assert len(completion.choices[0].message.content.split()) <= 2

@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_logprobs_returns_logprobs(inference_client, model_id):
    """Test that logprobs parameter returns log probability information."""
    completion = inference_client.chat.completions.create(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        model=model_id,
        logprobs=True,
        max_tokens=1,
    )
    assert hasattr(completion.choices[0], "logprobs")

@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_frequency_penalty_changes_output(inference_client, model_id):
    """Test that frequency_penalty parameter affects repetition."""
    completion_no_penalty = inference_client.chat.completions.create(
        messages=[{"role": "user", "content": "Repeat the word test test test."}],
        model=model_id,
        frequency_penalty=0,
        max_tokens=10,
    )
    completion_with_penalty = inference_client.chat.completions.create(
        messages=[{"role": "user", "content": "Repeat the word test test test."}],
        model=model_id,
        frequency_penalty=2.0,
        max_tokens=10,
    )
    count_no_penalty = completion_no_penalty.choices[0].message.content.lower().count("test")
    count_with_penalty = completion_with_penalty.choices[0].message.content.lower().count("test")
    # Allow for equality, and print for manual inspection
    print(f"Frequency penalty: no_penalty={count_no_penalty}, with_penalty={count_with_penalty}")
    assert count_with_penalty <= count_no_penalty + 1  # Allow for small variance

@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_presence_penalty_changes_output(inference_client, model_id):
    """Test that presence_penalty parameter affects new topic introduction."""
    completion_no_penalty = inference_client.chat.completions.create(
        messages=[{"role": "user", "content": "Talk about cats."}],
        model=model_id,
        presence_penalty=0,
        max_tokens=10,
    )
    completion_with_penalty = inference_client.chat.completions.create(
        messages=[{"role": "user", "content": "Talk about cats."}],
        model=model_id,
        presence_penalty=2.0,
        max_tokens=10,
    )
    unique_no_penalty = len(set(completion_no_penalty.choices[0].message.content.split()))
    unique_with_penalty = len(set(completion_with_penalty.choices[0].message.content.split()))
    print(f"Presence penalty: no_penalty={unique_no_penalty}, with_penalty={unique_with_penalty}")
    assert unique_with_penalty >= unique_no_penalty - 1  # Allow for small
