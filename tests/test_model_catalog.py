import sys
from pathlib import Path

import pytest
from pydantic import JsonValue

from jevpilot.model_catalog import list_chatgpt_models
from jevpilot.model_client import ProviderError


def fake_app_server(tmp_path: Path, pages: dict[str, JsonValue]) -> tuple[str, ...]:
    """Run a local JSON-RPC process with the Codex app-server line protocol."""
    script = tmp_path / "fake_app_server.py"
    script_source = (
        "import json, os, sys\n"
        f"expected_home = {str(tmp_path / 'codex')!r}\n"
        f"pages = {pages!r}\n"
        "assert os.environ['CODEX_HOME'] == expected_home\n"
        "assert 'OPENAI_API_KEY' not in os.environ\n"
        "assert 'CODEX_ACCESS_TOKEN' not in os.environ\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    method = request.get('method')\n"
        "    if method == 'initialize':\n"
        "        response = {'id': request['id'], 'result': {'userAgent': 'fake'}}\n"
        "    elif method == 'model/list':\n"
        "        cursor = request['params'].get('cursor') or 'start'\n"
        "        response = {'id': request['id'], 'result': pages[cursor]}\n"
        "    else:\n"
        "        continue\n"
        "    print(json.dumps(response), flush=True)\n"
    )
    _ = script.write_text(script_source, encoding="utf-8")
    return sys.executable, str(script)


@pytest.mark.asyncio
async def test_model_catalog_paginates_and_omits_hidden_models(tmp_path: Path) -> None:
    # Given
    command = fake_app_server(
        tmp_path,
        {
            "start": {
                "data": [
                    {
                        "id": "gpt-6.1-sol",
                        "model": "gpt-6.1-sol",
                        "displayName": "GPT-6.1 Sol",
                        "hidden": False,
                        "isDefault": True,
                        "description": "GPT-6.1 Sol",
                        "defaultReasoningEffort": "medium",
                        "serviceTiers": [
                            {
                                "id": "priority",
                                "name": "Fast",
                                "description": "2x speed, increased usage",
                            }
                        ],
                        "supportedReasoningEfforts": [
                            {"reasoningEffort": "low", "description": "Lighter reasoning"},
                            {"reasoningEffort": "medium", "description": "Balanced reasoning"},
                            {"reasoningEffort": "high", "description": "Deep reasoning"},
                            {"reasoningEffort": "xhigh", "description": "Extra-high reasoning"},
                            {"reasoningEffort": "max", "description": "Maximum reasoning"},
                            {"reasoningEffort": "ultra", "description": "Maximum delegation"},
                        ],
                    },
                    {
                        "id": "internal-hidden",
                        "model": "internal-hidden",
                        "displayName": "Hidden",
                        "hidden": True,
                        "isDefault": False,
                        "supportedReasoningEfforts": [],
                    },
                ],
                "nextCursor": "page-2",
            },
            "page-2": {
                "data": [
                    {
                        "id": "gpt-6.1-luna",
                        "model": "gpt-6.1-luna",
                        "displayName": "GPT-6.1 Luna",
                        "hidden": False,
                        "isDefault": False,
                        "description": "GPT-6.1 Luna",
                        "defaultReasoningEffort": "medium",
                        "serviceTiers": [
                            {
                                "id": "standard",
                                "name": "Standard",
                                "description": "Standard responses",
                            }
                        ],
                        "supportedReasoningEfforts": [
                            {"reasoningEffort": "low", "description": "Lighter reasoning"},
                            {"reasoningEffort": "medium", "description": "Balanced reasoning"},
                            {"reasoningEffort": "high", "description": "Deep reasoning"},
                        ],
                    }
                ],
                "nextCursor": None,
            },
        },
    )

    # When
    models = await list_chatgpt_models(command=command, profile=tmp_path / "codex")

    # Then
    assert [model.model_ref for model in models] == [
        "chatgpt-subscription/gpt-6.1-sol",
        "chatgpt-subscription/gpt-6.1-luna",
    ]
    assert models[0].is_default is True
    assert models[0].supports_fast is True
    assert models[0].reasoning_efforts == ("low", "medium", "high", "xhigh", "max")
    assert models[1].supports_fast is False
    assert models[1].reasoning_efforts == ("low", "medium", "high")


@pytest.mark.asyncio
async def test_model_catalog_rejects_malformed_server_data(tmp_path: Path) -> None:
    # Given
    command = fake_app_server(
        tmp_path,
        {"start": {"data": [{"id": "broken"}], "nextCursor": None}},
    )

    # When / Then
    with pytest.raises(ProviderError, match="could not read models"):
        _ = await list_chatgpt_models(command=command, profile=tmp_path / "codex")


@pytest.mark.asyncio
async def test_model_catalog_converts_process_start_error_to_provider_error(
    tmp_path: Path,
) -> None:
    # Given
    missing_codex = tmp_path / "missing-codex"

    # When / Then
    with pytest.raises(ProviderError, match="could not read models"):
        _ = await list_chatgpt_models(
            command=(str(missing_codex),),
            profile=tmp_path / "codex",
        )
