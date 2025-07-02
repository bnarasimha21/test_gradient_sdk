import os
import pytest
from gradientai import GradientAI
from dotenv import load_dotenv

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

@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_streaming_chat_completion_returns_choices(model_id):
    """Test that streaming chat completion yields choices for each model."""
    client = GradientAI(
        inference_key=os.environ.get("GRADIENTAI_API_KEY"),
    )
    stream = client.agents.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ],
        model=model_id,
        stream=True,
        max_tokens=5,
    )
    found = False
    for completion in stream:
        assert hasattr(completion, "choices")
        assert len(completion.choices) > 0
        assert hasattr(completion.choices[0], "delta")
        assert hasattr(completion.choices[0].delta, "content")
        assert completion.choices[0].delta.content is not None
        found = True
    assert found, "No streamed completions were received." 