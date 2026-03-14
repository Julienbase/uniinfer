"""Tests for model aliases."""

from __future__ import annotations

from uniinfer.models.aliases import (
    MODEL_ALIASES,
    get_alias_info,
    list_aliases,
    resolve_alias,
)


class TestResolveAlias:
    def test_known_alias_resolves(self) -> None:
        result = resolve_alias("mistral-7b")
        assert result == "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

    def test_alias_case_insensitive(self) -> None:
        result = resolve_alias("Mistral-7B")
        assert result == "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

    def test_unknown_model_passthrough(self) -> None:
        result = resolve_alias("some-org/some-model-GGUF")
        assert result == "some-org/some-model-GGUF"

    def test_local_path_passthrough(self) -> None:
        result = resolve_alias("/path/to/model.gguf")
        assert result == "/path/to/model.gguf"

    def test_tinyllama_alias(self) -> None:
        result = resolve_alias("tinyllama-1b")
        assert "TinyLlama" in result


class TestGetAliasInfo:
    def test_known_alias_returns_info(self) -> None:
        info = get_alias_info("mistral-7b")
        assert info is not None
        assert info.param_count_billions == 7.24
        assert info.display_name == "Mistral 7B Instruct v0.2"

    def test_unknown_returns_none(self) -> None:
        info = get_alias_info("nonexistent-model")
        assert info is None

    def test_alias_has_required_fields(self) -> None:
        for name, alias in MODEL_ALIASES.items():
            assert alias.repo_id, f"Alias '{name}' missing repo_id"
            assert alias.display_name, f"Alias '{name}' missing display_name"
            assert alias.param_count_billions > 0, f"Alias '{name}' has invalid param count"
            assert alias.default_quant, f"Alias '{name}' missing default_quant"
            assert alias.default_context_length > 0, f"Alias '{name}' has invalid context length"


class TestListAliases:
    def test_returns_sorted_by_param_count(self) -> None:
        aliases = list_aliases()
        param_counts = [a.param_count_billions for _, a in aliases]
        assert param_counts == sorted(param_counts)

    def test_returns_all_aliases(self) -> None:
        aliases = list_aliases()
        assert len(aliases) == len(MODEL_ALIASES)

    def test_contains_tinyllama(self) -> None:
        aliases = list_aliases()
        names = [name for name, _ in aliases]
        assert "tinyllama-1b" in names
