import asyncio
import os
import csv
import uuid
from enum import Enum
from typing import List, Union
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()

class LLMProvider(Enum):
    CHATGPT = "chatgpt"
    DEEPSEEK = "deepseek"

class Statements:
    def __init__(self, topic: str, count: int = 3):
        self.topic = topic
        self.count = count
        self.datasets = []

        self.request_count = {'count': 0}
        self.rate_limit_lock = asyncio.Lock()

        self.gpt_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.deepseek_client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )

    async def generate(self,
                       temperature: float=1,
                       top_p: float=0.9,
                       max_tokens: int=400,
                       provider: Union[LLMProvider, str] = LLMProvider.CHATGPT) -> List[str]:
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())

        client = self.gpt_client if provider == LLMProvider.CHATGPT else self.deepseek_client
        model = "gpt-4" if provider == LLMProvider.CHATGPT else "deepseek-reasoner"
        all_statements = []

        async def generate_single_statement():
            try:

                await self.rate_limit(self.request_count, self.rate_limit_lock)

                prompt = [
                        {"role": "system", "content": "You are a EU parliament representative."},
                        {"role": "user", "content": (
                            f"[Unique ID: {str(uuid.uuid4())}] "
                            f"Generate one unique debate contribution for the following topic: {self.topic}. "
                            "Return only the statement without any line-shifts, special signs or additional information."
                        )}
                ]

                response = await client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                
                content = response.choices[0].message.content.strip()
                statements = [s.strip() for s in content.split("\n") if s.strip()]
                
                self.datasets.extend([
                    {
                        'prompt': prompt,
                        'provider': provider.value,
                        'temperature': temperature,
                        'top_p': top_p,
                        'max_tokens': max_tokens,
                        'statement': statement
                    } 
                    for statement in statements
                ])
                return statements
                
            except Exception as e:
                print(f"Error generating statement: {str(e)}")
                return []

        tasks = [generate_single_statement() for _ in range(self.count)]
        for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f'{model}'):
            all_statements.extend(await result)

        return all_statements

    def save(self, filename: str = "statements.csv") -> None:
        if not self.datasets:
            print("No statements to save")
            return
        try:
            path = Path(filename)
            headers = ['prompt', 'provider', 'temperature', 'top_p', 'max_tokens', 'statement']
            with path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not path.exists():
                    writer.writeheader()
                writer.writerows(self.datasets)
        except Exception as e:
            raise Exception(f"Error saving dataset to CSV: {str(e)}")

    @staticmethod
    async def rate_limit(request_count: dict, lock: asyncio.Lock, max_requests: int = 50, pause_duration: int = 60):
        async with lock:
            if request_count['count'] >= max_requests:
                await asyncio.sleep(pause_duration)
                request_count['count'] = 0
            await asyncio.sleep(0.01)
            request_count['count'] += 1


async def main():
    statements = Statements(
        topic="A EU response to the 2025 Trump administrations trade measures, and global trade opportunities for the EU",
        count=10
    )
    await statements.generate(provider="chatgpt")
    await statements.generate(provider="deepseek")
    statements.save()

if __name__ == "__main__":
    asyncio.run(main())