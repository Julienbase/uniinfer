"""Tests for top-level uniinfer.chat() and chat_stream() functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from uniinfer.backends.interface import GenerationResult, StreamChunk


def _make_mock_engine(
    chat_result: GenerationResult | None = None,
    chat_stream_result: list[StreamChunk] | None = None,
    chat_side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock Engine instance."""
    mock = MagicMock()
    if chat_result:
        mock.chat.return_value = chat_result
    if chat_stream_result:
        mock.chat_stream.return_value = iter(chat_stream_result)
    if chat_side_effect:
        mock.chat.side_effect = chat_side_effect
    return mock


class TestUniinferChat:
    def test_chat_returns_text(self) -> None:
        mock_engine = _make_mock_engine(
            chat_result=GenerationResult(
                text="Hello there!", prompt_tokens=5, completion_tokens=3, total_tokens=8,
            )
        )

        with patch("uniinfer.engine.engine.Engine", return_value=mock_engine) as mock_cls:
            import uniinfer
            result = uniinfer.chat("mistral-7b", "Hello")

        assert result == "Hello there!"
        mock_engine.chat.assert_called_once()
        mock_engine.close.assert_called_once()

    def test_chat_with_system_prompt(self) -> None:
        mock_engine = _make_mock_engine(
            chat_result=GenerationResult(
                text="I am helpful.", prompt_tokens=10, completion_tokens=3, total_tokens=13,
            )
        )

        with patch("uniinfer.engine.engine.Engine", return_value=mock_engine):
            import uniinfer
            uniinfer.chat("mistral-7b", "Who are you?", system="You are a helpful bot.")

        call_args = mock_engine.chat.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful bot."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Who are you?"

    def test_chat_passes_engine_kwargs(self) -> None:
        mock_engine = _make_mock_engine(
            chat_result=GenerationResult(
                text="ok", prompt_tokens=1, completion_tokens=1, total_tokens=2,
            )
        )

        with patch("uniinfer.engine.engine.Engine", return_value=mock_engine) as mock_cls:
            import uniinfer
            uniinfer.chat("mistral-7b", "Hi", device="cpu", temperature=0.5)

        mock_cls.assert_called_once_with(model="mistral-7b", device="cpu")

    def test_chat_closes_engine_on_error(self) -> None:
        mock_engine = _make_mock_engine(chat_side_effect=RuntimeError("generation failed"))

        with patch("uniinfer.engine.engine.Engine", return_value=mock_engine):
            import uniinfer
            try:
                uniinfer.chat("mistral-7b", "Hello")
            except RuntimeError:
                pass

        mock_engine.close.assert_called_once()


class TestUniinferChatStream:
    def test_chat_stream_yields_chunks(self) -> None:
        mock_engine = _make_mock_engine(
            chat_stream_result=[
                StreamChunk(text="Hello", finished=False),
                StreamChunk(text=" world", finished=True),
            ]
        )

        with patch("uniinfer.engine.engine.Engine", return_value=mock_engine):
            import uniinfer
            chunks = list(uniinfer.chat_stream("mistral-7b", "Hi"))

        assert len(chunks) == 2
        assert chunks[0].text == "Hello"
        assert chunks[1].finished is True
        mock_engine.close.assert_called_once()
