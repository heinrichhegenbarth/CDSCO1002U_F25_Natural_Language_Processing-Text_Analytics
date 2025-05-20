import asyncio
import os
import threading
import time
import csv
import uuid
from enum import Enum
from typing import List, Union
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm
from queue import Queue
import threading

load_dotenv()

class LLMProvider(Enum):
    CHATGPT = "chatgpt"
    DEEPSEEK = "deepseek"

class Statements:
    def __init__(self, topic: str, count: int = 3):
        self.topic = topic
        self.count = count
        self.datasets = []
        
        self.gpt_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.deepseek_client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )

    async def generate(self,
                       temperature: float=1.5,
                       max_tokens: int=400,
                       provider: Union[LLMProvider, str] = LLMProvider.CHATGPT) -> List[str]:
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())

        client = self.gpt_client if provider == LLMProvider.CHATGPT else self.deepseek_client
        model = "gpt-4" if provider == LLMProvider.CHATGPT else "deepseek-reasoner"
        all_statements = []

        start_time = time.time()

        for _ in tqdm(range(self.count), desc=f"{model}"):
            try:

                prompt = [
                        {"role": "system", "content": "You are a EU parliament representative."},
                        {"role": "user", "content": (
                            f"[Unique ID: {str(uuid.uuid4())}]"
                            f"Generate one unique debate contribution for the following topic: {self.topic}. "
                            "Make the statement as if you were a representative of the EU parliament. "
                            "Return only the statement without any line-shifts or additional information."
                        )}
                ]

                response = await client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                content = response.choices[0].message.content.strip()
                statements = [s.strip() for s in content.split("\n") if s.strip()]
                
                self.datasets.extend([
                    {
                        'prompt': prompt,
                        'provider': provider.value,
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'statement': statement
                    } 
                    for statement in statements
                ])
                all_statements.extend(statements)
                
            except Exception as e:
                print(f"Error generating statement: {str(e)}")

        end_time = time.time()
        print(f"Duration {model}: {end_time - start_time:.4f} seconds")

        return all_statements

    def save(self, filename: str = "statements.csv") -> None:
        if not self.datasets:
            print("No statements to save")
            return

        try:
            path = Path(filename)
            headers = ['prompt', 'provider', 'temperature', 'max_tokens', 'statement']
            
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.datasets)
                
            print(f"Successfully saved {len(self.datasets)} statements to {path}")
            
        except Exception as e:
            raise Exception(f"Error saving dataset to CSV: {str(e)}")

async def main():
    statements = Statements(
        topic="A unified EU response to unjustified US trade measures and global trade opportunities for the EU",
        count=10
    )
    await statements.generate(provider="chatgpt")
    await statements.generate(provider="deepseek")
    statements.save()

def call_llm(topic, temperature, max_tokens):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = "gpt-4"

    prompt = [
        {"role": "system", "content": "You are a EU parliament representative."},
        {"role": "user", "content": (
            f"[Unique ID: {str(uuid.uuid4())}]"
            f"Generate one unique debate contribution for the following topic: {topic}. "
            "Make the statement as if you were a representative of the EU parliament. "
            "Return only the statement without any line-shifts or additional information."
        )}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content.strip()
    statements = [s.strip() for s in content.split("\n") if s.strip()]

    return [
        {
            'prompt': prompt,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'statement': statement
        }
        for statement in statements
    ]

for _ in range(3):
    call_llm(temperature=1.5,
             max_tokens=400,
             topic="A unified EU response to unjustified US trade measures and global trade opportunities for the EU")

#if __name__ == "__main__":
#    asyncio.run(main())