import asyncio
import time
import os
import json
from enum import Enum
from typing import List, Dict, Optional, Union
from datetime import datetime
import random
from dotenv import load_dotenv

from openai import AsyncOpenAI

load_dotenv()

class LLMProvider(Enum):
    CHATGPT = "chatgpt"
    DEEPSEEK = "deepseek"

class Statements:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.datasets = {}

        self.gpt_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        #try:
        #    self.deepseek_client = DeepSeekAPI(configs["deepseek_api_key"])
        #except Exception as e:
        #    print(e)
        #    self.deepseek_client = None

    async def get(self,
                  count: int = 1,
                  provider: Union[LLMProvider, str] = LLMProvider.CHATGPT) -> List[str]:

        for i in range(0, 1):

            if isinstance(provider, str):
                provider = LLMProvider(provider.lower())

            temperature = round(random.uniform(0.5, 0.9), 2)
            max_tokens = random.randint(200, 500)

            formatted_prompt = \
                f"""
                Generate a unique debate contributions for the following topic: {self.prompt}.
                Make each statement as if you where a representative of the EU parliament.
                Return only the statements, one per line.
                """

            try:
                statements = await self._generate_statements(formatted_prompt, temperature, max_tokens, provider)

                self.datasets[f'instance_{i}'] = {
                    'timestamp': datetime.now().isoformat(),
                    'prompt': self.prompt,
                    'provider': provider.value,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'statements': statements,
                    'count_requested': count,
                    'count_generated': len(statements)
                }

                return statements
            except Exception as e:
                raise Exception(f"Error generating statements: {str(e)}")
        return None

    def save_statements(self):
        try:
            for name, dataset in self.datasets.items():
                with open(f'{name}.json', 'w') as f:
                    json.dump(dataset, f, indent=2)
        except Exception as e:
            raise Exception(f"Error saving dataset: {str(e)}")

    async def _generate_statements(self, prompt: str, temperature: float, max_tokens: int, provider: LLMProvider) -> List[str]:

        try:

            if provider == LLMProvider.CHATGPT:
                client = self.gpt_client
                model = "gpt-3.5-turbo"
            else:
                client = self.deepseek_client
                model = "deepseek-chat"

            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ]

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            statements = response.choices[0].message.content.strip().split("\n")
            return [s.strip() for s in statements if s.strip()]

        except Exception as e:
            raise Exception(f"{provider.value.title()} API error: {str(e)}")

if __name__ == "__main__":
    gen = Statements(
        prompt="A unified EU response to unjustified US trade measures and global trade opportunities for the EU",
    )
    asyncio.run(gen.get(count=3, provider="chatgpt"))
    gen.save_statements()
