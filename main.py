import os
from gradientai import GradientAI
from dotenv import load_dotenv

load_dotenv()   

inference_client = GradientAI(
    inference_key=os.environ.get("GRADIENTAI_API_KEY"),  # This is the default and can be omitted
)

completion = inference_client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
    model="llama3-70b-instruct",
    temperature=0.7,
    max_tokens=10        
)

print(completion.choices[0].message)
