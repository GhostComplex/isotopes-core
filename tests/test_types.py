"""Tests for isotopes-core types."""

from isotopes_core import (
    TextContent,
    ImageContent,
    UserMessage,
    AssistantMessage,
    Context,
    Usage,
)


class TestTextContent:
    def test_create(self):
        content = TextContent(text="hello")
        assert content.text == "hello"
        assert content.type == "text"

    def test_frozen(self):
        content = TextContent(text="hello")
        try:
            content.text = "world"  # type: ignore
            assert False, "Should be frozen"
        except Exception:
            pass


class TestUserMessage:
    def test_text_helper(self):
        msg = UserMessage.text("hello")
        assert msg.role == "user"
        assert len(msg.content) == 1
        assert msg.content[0].text == "hello"


class TestAssistantMessage:
    def test_text_helper(self):
        msg = AssistantMessage.text("hello")
        assert msg.role == "assistant"
        assert len(msg.content) == 1
        assert msg.content[0].text == "hello"


class TestContext:
    def test_add_message(self):
        ctx = Context(system_prompt="You are helpful.")
        ctx.add_user_message("Hello")
        ctx.add_assistant_message("Hi!")
        assert len(ctx.messages) == 2

    def test_clear(self):
        ctx = Context()
        ctx.add_user_message("Hello")
        ctx.clear()
        assert len(ctx.messages) == 0


class TestUsage:
    def test_total_tokens(self):
        usage = Usage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_add(self):
        u1 = Usage(input_tokens=100, output_tokens=50)
        u2 = Usage(input_tokens=200, output_tokens=100)
        total = u1 + u2
        assert total.input_tokens == 300
        assert total.output_tokens == 150
