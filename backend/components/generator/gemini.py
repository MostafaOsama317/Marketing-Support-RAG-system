import os
from dotenv import load_dotenv
from wasabi import msg
from goldenverba.components.interfaces import Generator

try:
    from openai import AsyncOpenAI
except ImportError:
    pass

load_dotenv()


class OpenRouterGemmaGenerator(Generator):
    """
    Generator using Google's Gemma model via OpenRouter.
    """

    def __init__(self):
        super().__init__()
        self.name = "OpenRouter Gemma"
        self.description = "Generator using google/gemma-4-31b-it:free via OpenRouter"
        self.requires_library = ["openai"]
        self.requires_env = [
            "OPENROUTER_API_KEY",
        ]
        self.streamable = True
        self.model_name = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
        self.context_window = 8192

    async def generate_stream(
        self,
        queries: list[str],
        context: list[str],
        conversation: dict = None,
    ):
        """Generate a stream of response dicts based on a list of queries and list of contexts."""
        
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            yield {
                "message": "Missing OPENROUTER_API_KEY",
                "finish_reason": "stop",
            }
            return

        if conversation is None:
            conversation = {}
            
        messages = self.prepare_messages(queries, context, conversation)

        try:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            stream = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
            )

            async for chunk in stream:
                if len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    content = choice.delta.content or ""
                    finish_reason = choice.finish_reason or ""

                    # نفس الصرامة في التعامل مع فلاتر الأمان
                    if finish_reason == "content_filter":
                        yield {
                            "message": " < Canceled due SAFETY REASONS >",
                            "finish_reason": "stop",
                        }
                        break
                    
                    if content or finish_reason:
                        yield {
                            "message": content,
                            "finish_reason": finish_reason if finish_reason != "null" else "",
                        }

        except Exception as e:
            raise e

    def prepare_messages(
        self, queries: list[str], context: list[str], conversation: list
    ):
        """Prepares strict alternating messages formatted for OpenRouter / OpenAI structure."""
        messages = []

        for message in conversation:
            # تحويل الأدوار للي بيفهمه OpenRouter (assistant بدل model)
            role = "assistant" if message.type == "model" else message.type
            messages.append({"role": role, "content": message.content})

        query = " ".join(queries)
        user_context = " ".join(context)

        # نفس صيغة الـ Prompt الصارمة
        prompt_content = (
            f"{user_context} Please answer this query: '{query}' with this provided context. "
            "Only use the context if it is necessary to answer the question."
        )

        messages.append({"role": "user", "content": prompt_content})

        # تطبيق نفس نظام الحماية وتنظيف المحادثة
        messages = self.ensure_user_model_alteration(messages)

        return messages

    def ensure_user_model_alteration(self, messages: list[dict]) -> list[dict]:
        """
        نفس اللوجيك الصارم:
        1. منع الـ system roles وتحويلها لـ assistant.
        2. التأكد إن المحادثة بتبدأ بـ user.
        3. منع تكرار نفس الدور ورا بعضه (User->Assistant->User).
        """
        for message in messages:
            if message["role"] == "system":
                message["role"] = "assistant"
            # توحيد مسمى الموديل لـ assistant عشان متوافق مع OpenRouter
            elif message["role"] == "model":
                message["role"] = "assistant"

        # لازم تبدأ بـ user
        while len(messages) > 0 and messages[0]["role"] == "assistant":
            messages = messages[1:]

        if not messages:
            return []

        new_messages = []
        current_role = ""

        for message in messages:
            if message["role"] == current_role:
                # لو نفس الدور اتكرر ورا بعض، بيستبدل القديم بالجديد (نفس سلوك كودك الأصلي بالظبط)
                new_messages[-1] = message
            else:
                new_messages.append(message)
                current_role = message["role"]

        return new_messages