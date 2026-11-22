"""#0: Inkling native calls (<|content_invoke_tool_json|> JSON) must reach the tool client with exact argument values.

The rewriter translates the model's JSON call into the <parameter> dialect, which reads declared-string
values as raw text, so a string value is written raw (never json.dumps'd: that would wrap it in quotes,
the tool client would see "\"https://...\"", and the re-rendered history would teach the model to escape
harder each turn).  Non-string values keep their JSON encoding and are decoded by the parameter types.

    python -m unittest serve.test_inkling_call_values -v
"""
import json
import unittest

from serve.frontend import InklingOutputParser, InklingRewriter

SCHEMA = {"name": "webfetch", "parameters": {"properties": {
    "url": {"type": "string"},
    "prompt": {"type": "string"},
    "limit": {"type": "integer"},
    "raw": {"type": "boolean"},
    "tags": {"type": "array"},
}}}


def call_text(args: dict) -> str:
    return ("<|message_model|>webfetch<|content_invoke_tool_json|>" +
            json.dumps({"name": "webfetch", "args": args}, separators=(",", ":")) + "<|end_message|>")


def parse(text: str, schema=SCHEMA, stream=False):
    parser = InklingOutputParser(thinking=False, tools=[schema], stream_tools=stream)
    events = []
    for i in range(0, len(text), 7):        # a mark split across deltas is held until it is whole
        events.extend(parser.feed(text[i:i + 7]))
    events.extend(parser.finish())
    calls = [e.call for e in events if e.kind == "tool_call"]
    return calls, events


class InklingCallValues(unittest.TestCase):
    def test_string_values_arrive_without_quotes(self):
        args = {"url": "https://example.com/skills/clawhub-ai-simmer", "prompt": "summarize the page", "limit": 5}
        calls, _ = parse(call_text(args))
        self.assertEqual([c.arguments for c in calls], [args])

    def test_empty_and_edge_strings(self):
        args = {"url": "", "prompt": "a < b and </parameter> lookalikes stay", "limit": 0}
        calls, _ = parse(call_text(args))
        self.assertEqual(calls[0].arguments["url"], "")
        self.assertEqual(calls[0].arguments["prompt"], args["prompt"])
        self.assertEqual(calls[0].arguments["limit"], 0)

    def test_non_string_values_are_decoded(self):
        args = {"url": "https://example.com/", "prompt": "p", "limit": 10, "raw": True, "tags": ["a", "b"]}
        calls, _ = parse(call_text(args))
        self.assertEqual(calls[0].arguments, args)

    def test_streamed_arguments_json_is_clean(self):
        text = call_text({"url": "https://example.com/", "prompt": "summarize"})
        calls, events = parse(text, stream=True)
        self.assertEqual([c.arguments for c in calls],
                         [{"url": "https://example.com/", "prompt": "summarize"}])
        streamed = "".join(e.text for e in events if e.kind == "tool_args")
        self.assertEqual(json.loads(streamed), {"url": "https://example.com/", "prompt": "summarize"})

    def test_rewriter_writes_strings_raw(self):
        r = InklingRewriter(False)
        out = r.feed(call_text({"url": "https://example.com/"})) + r.finish()
        self.assertIn("<parameter=url>\nhttps://example.com/\n</parameter>", out)
        self.assertNotIn('"https://', out)


if __name__ == "__main__":
    unittest.main()
