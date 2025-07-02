import os
import pytest
import asyncio
from gradientai import AsyncGradientAI
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

@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", MODEL_IDS)
async def test_async_chat_completion_returns_choices(model_id):
    """Test that an async chat completion returns choices for each model."""
    client = AsyncGradientAI(
        api_key=os.environ.get("GRADIENTAI_API_KEY"),
    )
    completion = await client.agents.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ],
        model=model_id,
        max_tokens=5,
    )
    assert hasattr(completion, "choices")
    assert len(completion.choices) > 0
    assert hasattr(completion.choices[0], "message") 