#!/usr/bin/env python3
"""Contract tests for the hosted VLM adapter; no network calls."""

from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from agent import OPENAI_SYSTEM, SYSTEM, OpenAIVLMAgent


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        usage = SimpleNamespace(
            prompt_tokens=1609,
            completion_tokens=32,
            prompt_tokens_details=SimpleNamespace(cached_tokens=100))
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=usage)


def fake_agent(content='{"instances":[]}'):
    completions = FakeCompletions(content)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions))
    return OpenAIVLMAgent(client=client), completions


def test_openai_preserves_original_resolution_and_uses_json_mode():
    agent, completions = fake_agent()
    image = Image.new("RGB", (1920, 640), "navy")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Return JSON only."},
        ]},
    ]

    raw = agent._gen(
        messages, [image], max_new_tokens=300,
        label="inspect_domain_view_atomic")

    assert raw == '{"instances":[]}'
    request = completions.requests[0]
    assert request["model"] == "gpt-5.6-sol"
    assert request["reasoning_effort"] == "medium"
    assert request["max_completion_tokens"] >= 2048
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][0]["content"] == OPENAI_SYSTEM
    image_part = request["messages"][1]["content"][0]
    assert image_part["image_url"]["detail"] == "auto"
    encoded = image_part["image_url"]["url"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as decoded:
        assert decoded.size == (1920, 640)
    assert agent.trace[0]["img_size"] == [[1920, 640]]
    assert agent.trace[0]["in_tokens"] == 1609
    assert agent.trace[0]["estimated_cost_usd"] > 0


def test_batched_domain_views_are_one_call_with_stable_image_indices():
    response = ('{"instances":[{"image_index":1,'
                '"bbox_norm":[100,200,300,400],'
                '"description":"red pillow","confidence":0.95,'
                '"sam_queries":["red pillow"]}]}')
    agent, completions = fake_agent(response)
    views = [
        ({"domain_kind": "floor", "reason": "visible floor"},
         Image.new("RGB", (800, 600), "red")),
        ({"domain_kind": "floor", "reason": "visible floor"},
         Image.new("RGB", (800, 600), "blue")),
    ]

    value, raw = agent.inspect_domain_views_atomic(
        views, "pillow", "How many pillows are on the floor?", "room story")

    assert value["instances"][0]["image_index"] == 1
    assert raw == response
    assert len(completions.requests) == 1
    content = completions.requests[0]["messages"][1]["content"]
    assert sum(part.get("type") == "image_url" for part in content) == 2
    prompt = next(part["text"] for part in content
                  if part.get("type") == "text")
    assert "partially visible object behind it" in prompt


def test_highlighted_candidate_crops_are_classified_in_one_call():
    response = ('{"results":['
                '{"image_index":0,"what_is_it":"white pillow",'
                '"is_class":true,"color":"white",'
                '"distinguishing_marks":"plain","confidence":0.98},'
                '{"image_index":1,"what_is_it":"gray pillow",'
                '"is_class":true,"color":"gray",'
                '"distinguishing_marks":"plain","confidence":0.97}]}')
    agent, completions = fake_agent(response)
    requests = [
        {"crop": Image.new("RGB", (700, 600), color),
         "concept": "pillow", "tag": f"N{index}", "highlighted": True}
        for index, color in enumerate(("white", "gray"), start=1)]

    results = agent.inspect_crops_batch(requests, tag="capture_00")

    assert [item["what_is_it"] for item in results] == [
        "white pillow", "gray pillow"]
    assert all(item["is_class"] is True for item in results)
    assert len(completions.requests) == 1
    content = completions.requests[0]["messages"][1]["content"]
    assert sum(part.get("type") == "image_url" for part in content) == 2


def test_high_reasoning_reserves_hidden_reasoning_tokens():
    completions = FakeCompletions('{"instances":[]}')
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent = OpenAIVLMAgent(client=client, reasoning_effort="high")
    agent._gen(
        [{"role": "user", "content": [
            {"type": "text", "text": "Return one JSON object."}]}],
        [], max_new_tokens=200, label="inspect_domain_view_atomic")
    assert completions.requests[0]["max_completion_tokens"] >= 4096
