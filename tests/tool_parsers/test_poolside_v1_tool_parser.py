# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tool_parsers.poolside_v1_tool_parser import PoolsideV1ToolParser


@pytest.fixture
def request():
    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        )
    ]
    return ChatCompletionRequest(
        messages=[],
        model="test-model",
        tools=tools,
        tool_choice="auto",
    )


@pytest.mark.parametrize(
    "model_output",
    [
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value></tool_call>",
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n"
        "<arg_value>Paris</arg_value>\n</tool_call>",
    ],
    ids=["one-line", "multiline"],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_extract_weather_call(default_tokenizer, request, model_output, streaming):
    parser = PoolsideV1ToolParser(default_tokenizer, tools=request.tools)
    content, tool_calls = run_tool_extraction(
        parser,
        model_output,
        request=request,
        streaming=streaming,
    )

    assert content is None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"city": "Paris"}
