from unittest.mock import MagicMock, patch

from corpus.retriever import PaperChunk


def _chunk(n=0):
    return PaperChunk(
        chunk_id=f"1706.03762-results-{n}", paper_id="1706.03762",
        title="Attention Is All You Need", authors=["Vaswani, A."], year=2017,
        journal="NeurIPS 2017", section="results",
        text="Our model achieves 28.4 BLEU on English-German translation.",
        score=0.91, source_url="https://arxiv.org/abs/1706.03762",
    )


def _mock_llm_response(text="test response"):
    r = MagicMock()
    r.content = [MagicMock(text=(
        f'{{"content": "{text}", "citations": [], "confidence": 0.8, "dissent": null}}'
    ))]
    r.stop_reason = "end_turn"
    r.usage = MagicMock(input_tokens=100, output_tokens=50)
    return r


def _make_agent(retriever=None):
    with patch.object(
        __import__("agents.base_agent", fromlist=["DomainAgent"]).DomainAgent,
        "_load_persona", return_value="test persona"
    ):
        from agents.base_agent import DomainAgent
        client = MagicMock()
        client.messages.create.return_value = _mock_llm_response()
        return DomainAgent("ai_ml_systems", client, retriever=retriever), client


def test_agent_injects_context_block_when_retriever_returns_chunks():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [_chunk(i) for i in range(3)]

    agent, client = _make_agent(retriever=mock_retriever)
    agent.respond("What is attention?", [])

    message_content = client.messages.create.call_args[1]["messages"][0]["content"]
    assert "[CONTEXT" in message_content
    assert "Attention Is All You Need" in message_content


def test_agent_omits_tools_when_retriever_returns_enough_chunks():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [_chunk(i) for i in range(3)]

    agent, client = _make_agent(retriever=mock_retriever)
    agent.respond("What is attention?", [])

    call_kwargs = client.messages.create.call_args[1]
    assert "tools" not in call_kwargs


def test_agent_falls_back_to_web_search_when_fewer_than_3_chunks():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [_chunk()]  # only 1

    agent, client = _make_agent(retriever=mock_retriever)
    agent.respond("What is attention?", [])

    call_kwargs = client.messages.create.call_args[1]
    tools = call_kwargs.get("tools", [])
    assert any(t.get("type") == "web_search_20250305" for t in tools)


def test_agent_uses_web_search_when_no_retriever():
    agent, client = _make_agent(retriever=None)
    agent.respond("What is attention?", [])

    call_kwargs = client.messages.create.call_args[1]
    tools = call_kwargs.get("tools", [])
    assert any(t.get("type") == "web_search_20250305" for t in tools)
