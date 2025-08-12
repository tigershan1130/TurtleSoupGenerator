import configparser
from typing import List, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp
import json

from langchain.schema import LLMResult
from langchain.schema.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.schema.output import ChatGeneration  # Needed for proper LLMResult structure

class DeepSeekLLM:
    def __init__(self, api_key: str, api_base: str, model_name: str, temperature: float, max_tokens: int):
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = None  # Delayed initialization

    async def init_session(self):
        """Initialize aiohttp session (must be called inside async context)"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

    def _format_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to DeepSeek API format"""
        formatted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
        return formatted

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResult:
        """Send async generation request to DeepSeek"""
        if self.session is None:
            raise RuntimeError("Session not initialized. Use await llm.init_session() before calling.")

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": self._format_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()

                if "error" in result:
                    raise Exception(f"API Error: {result['error'].get('message', 'Unknown error')}")

                content = result['choices'][0]['message']['content']
                token_usage = result.get('usage', {})

                return LLMResult(
                    generations=[[ChatGeneration(message=AIMessage(content=content))]],
                    llm_output={
                        'token_usage': token_usage,
                        'model_name': self.model_name
                    }
                )
        except aiohttp.ClientError as e:
            raise Exception(f"Network error: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {str(e)}") from e

    async def ainvoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Compatible LangChain interface: get final message only"""
        result = await self.agenerate(messages)
        return result.generations[0][0].message

    async def close(self):
        """Gracefully close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
