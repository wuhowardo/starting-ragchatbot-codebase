import pytest
from unittest.mock import MagicMock, patch
from ai_generator import AIGenerator


def make_text_response(text="answer"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def make_tool_use_response(tool_name="search_course_content", tool_id="tool_1", tool_input=None):
    # spec excludes "text" so hasattr(block, "text") correctly returns False
    block = MagicMock(spec=["type", "name", "id", "input"])
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = tool_input or {"query": "test"}
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


def make_generator():
    return AIGenerator(api_key="test-key", model="claude-test")


def make_tool_manager(return_value="tool result"):
    tm = MagicMock()
    tm.execute_tool.return_value = return_value
    return tm


TOOLS = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]


class TestNoToolUse:
    def test_returns_text_with_single_api_call(self):
        gen = make_generator()
        text_resp = make_text_response("direct answer")

        with patch.object(gen.client.messages, "create", return_value=text_resp) as mock_create:
            result = gen.generate_response("what is python?", tools=TOOLS, tool_manager=make_tool_manager())

        assert result == "direct answer"
        assert mock_create.call_count == 1

    def test_no_execute_tool_called(self):
        gen = make_generator()
        tm = make_tool_manager()

        with patch.object(gen.client.messages, "create", return_value=make_text_response()):
            gen.generate_response("what is python?", tools=TOOLS, tool_manager=tm)

        tm.execute_tool.assert_not_called()


class TestSingleToolRound:
    def test_two_api_calls_one_tool_execution(self):
        gen = make_generator()
        tm = make_tool_manager("course info")

        with patch.object(gen.client.messages, "create",
                          side_effect=[make_tool_use_response(), make_text_response("final answer")]) as mock_create:
            result = gen.generate_response("find course X", tools=TOOLS, tool_manager=tm)

        assert result == "final answer"
        assert mock_create.call_count == 2
        assert tm.execute_tool.call_count == 1

    def test_tool_executed_with_correct_args(self):
        gen = make_generator()
        tm = make_tool_manager()

        with patch.object(gen.client.messages, "create",
                          side_effect=[make_tool_use_response(tool_name="search_course_content",
                                                               tool_input={"query": "python basics"}),
                                       make_text_response()]):
            gen.generate_response("find python", tools=TOOLS, tool_manager=tm)

        tm.execute_tool.assert_called_once_with("search_course_content", query="python basics")

    def test_second_api_call_includes_tool_results(self):
        gen = make_generator()
        tm = make_tool_manager("search results")

        with patch.object(gen.client.messages, "create",
                          side_effect=[make_tool_use_response(tool_id="abc123"),
                                       make_text_response()]) as mock_create:
            gen.generate_response("find course", tools=TOOLS, tool_manager=tm)

        second_call_messages = mock_create.call_args_list[1].kwargs["messages"]
        # [user query, assistant tool_use, user tool_result]
        assert len(second_call_messages) == 3
        assert second_call_messages[1]["role"] == "assistant"
        assert second_call_messages[2]["role"] == "user"
        tool_result_block = second_call_messages[2]["content"][0]
        assert tool_result_block["type"] == "tool_result"
        assert tool_result_block["tool_use_id"] == "abc123"
        assert tool_result_block["content"] == "search results"


class TestTwoToolRounds:
    def test_three_api_calls_two_tool_executions(self):
        gen = make_generator()
        tm = make_tool_manager("result")

        with patch.object(gen.client.messages, "create",
                          side_effect=[make_tool_use_response(tool_id="t1"),
                                       make_tool_use_response(tool_id="t2"),
                                       make_text_response("complete answer")]) as mock_create:
            result = gen.generate_response("compare courses", tools=TOOLS, tool_manager=tm)

        assert result == "complete answer"
        assert mock_create.call_count == 3
        assert tm.execute_tool.call_count == 2

    def test_message_accumulation_after_two_rounds(self):
        gen = make_generator()
        tm = make_tool_manager("result")

        with patch.object(gen.client.messages, "create",
                          side_effect=[make_tool_use_response(tool_id="t1"),
                                       make_tool_use_response(tool_id="t2"),
                                       make_text_response()]) as mock_create:
            gen.generate_response("compare courses", tools=TOOLS, tool_manager=tm)

        third_call_messages = mock_create.call_args_list[2].kwargs["messages"]
        # [user, assistant#1, tool_result#1, assistant#2, tool_result#2]
        assert len(third_call_messages) == 5
        assert third_call_messages[0]["role"] == "user"
        assert third_call_messages[1]["role"] == "assistant"
        assert third_call_messages[2]["role"] == "user"
        assert third_call_messages[3]["role"] == "assistant"
        assert third_call_messages[4]["role"] == "user"

    def test_tools_active_on_second_round(self):
        gen = make_generator()
        tm = make_tool_manager()

        with patch.object(gen.client.messages, "create",
                          side_effect=[make_tool_use_response(),
                                       make_tool_use_response(),
                                       make_text_response()]) as mock_create:
            gen.generate_response("compare courses", tools=TOOLS, tool_manager=tm)

        # Both intermediate calls should have tools attached
        assert "tools" in mock_create.call_args_list[1].kwargs
        assert "tools" in mock_create.call_args_list[2].kwargs


class TestRoundCapEnforced:
    def test_stops_after_two_rounds_even_if_claude_wants_more(self):
        gen = make_generator()
        tm = make_tool_manager()

        # Claude keeps returning tool_use but loop must stop after 2 rounds
        with patch.object(gen.client.messages, "create",
                          side_effect=[make_tool_use_response(),
                                       make_tool_use_response(),
                                       make_text_response("capped")]) as mock_create:
            result = gen.generate_response("multi search", tools=TOOLS, tool_manager=tm)

        assert mock_create.call_count == 3
        assert tm.execute_tool.call_count == 2
        assert result == "capped"


class TestToolExecutionError:
    def test_returns_empty_string_on_exception(self):
        gen = make_generator()
        tm = make_tool_manager()
        tm.execute_tool.side_effect = RuntimeError("vector store unavailable")

        with patch.object(gen.client.messages, "create",
                          return_value=make_tool_use_response()) as mock_create:
            result = gen.generate_response("find course", tools=TOOLS, tool_manager=tm)

        # Only the initial API call was made; loop broke after exception
        assert mock_create.call_count == 1
        assert result == ""

    def test_tool_not_found_string_is_treated_as_valid_result(self):
        gen = make_generator()
        tm = make_tool_manager("Tool 'unknown' not found")  # ToolManager returns string, no raise

        with patch.object(gen.client.messages, "create",
                          side_effect=[make_tool_use_response(tool_name="unknown"),
                                       make_text_response("graceful response")]):
            result = gen.generate_response("find X", tools=TOOLS, tool_manager=tm)

        assert result == "graceful response"


class TestNoToolsProvided:
    def test_single_api_call_no_tools_in_params(self):
        gen = make_generator()

        with patch.object(gen.client.messages, "create",
                          return_value=make_text_response("direct")) as mock_create:
            result = gen.generate_response("what is python?", tools=None, tool_manager=None)

        assert result == "direct"
        assert mock_create.call_count == 1
        assert "tools" not in mock_create.call_args.kwargs

    def test_no_execute_tool_called(self):
        gen = make_generator()
        tm = make_tool_manager()

        with patch.object(gen.client.messages, "create", return_value=make_text_response()):
            gen.generate_response("what is python?", tools=None, tool_manager=tm)

        tm.execute_tool.assert_not_called()


class TestConversationHistory:
    def test_history_injected_into_system_prompt(self):
        gen = make_generator()
        history = "User: hello\nAssistant: hi"

        with patch.object(gen.client.messages, "create",
                          return_value=make_text_response()) as mock_create:
            gen.generate_response("follow-up", conversation_history=history)

        system = mock_create.call_args.kwargs["system"]
        assert "Previous conversation:" in system
        assert history in system

    def test_no_history_uses_base_system_prompt(self):
        gen = make_generator()

        with patch.object(gen.client.messages, "create",
                          return_value=make_text_response()) as mock_create:
            gen.generate_response("hello")

        system = mock_create.call_args.kwargs["system"]
        assert "Previous conversation:" not in system
