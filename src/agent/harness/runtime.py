"""Explicit model/tool loop with per-run budgets and redacted audit events.

Only selected local pure tools are exposed. Timeouts cannot kill a synchronous
thread; tools with side effects must not be added without a separate policy.
"""
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from uuid import uuid4

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ExecutionRun:
    model_limit: int
    tool_limit: int
    deadline_seconds: float
    tool_timeout: float
    input_limit: int
    output_limit: int
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    tool_calls: int = 0
    events: list = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    def remaining(self):
        remaining = self.deadline_seconds - (time.monotonic() - self.started)
        if remaining <= 0:
            raise BudgetExceeded("deadline")
        return remaining

    def event(self, kind, **data):
        self.events.append({"sequence": len(self.events), "kind": kind,
                            "elapsed_ms": round((time.monotonic() - self.started) * 1000, 3), **data})

    def report(self, status):
        return {"run_id": self.run_id, "status": status, "model_calls": self.model_calls,
                "tool_calls": self.tool_calls, "usage": self.usage, "events": self.events,
                "limits": {"model_calls": self.model_limit, "tool_calls": self.tool_limit,
                           "deadline_seconds": self.deadline_seconds}}


def create_run(settings):
    return ExecutionRun(settings.operation_max_model_calls, settings.operation_max_tool_calls,
                        settings.operation_deadline_seconds, settings.operation_tool_timeout_seconds,
                        settings.operation_max_input_chars, settings.operation_max_output_chars)


class BoundedToolAgent:
    """Reusable compiled capabilities; mutable budget belongs to the invocation."""
    def __init__(self, model, tools, prompt):
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate tool names")
        self.tools = {tool.name: tool for tool in tools}
        self.model = model.bind_tools(tools) if tools else model
        self.prompt = prompt

    async def ainvoke(self, state, config=None):
        run = (config or {}).get("configurable", {}).get("execution_run")
        if not isinstance(run, ExecutionRun):
            raise ValueError("Execution budget required")
        messages = [SystemMessage(content=self.prompt), *state.get("messages", [])]
        run.event("started", tools=sorted(self.tools))
        executed_call_ids = set()
        while True:
            run.remaining()
            if run.model_calls >= run.model_limit:
                raise BudgetExceeded("model_calls")
            # Fail explicitly instead of silently deleting source text or tool results.
            size = sum(len(str(message.content)) for message in messages)
            size += sum(len(json.dumps(getattr(message, "tool_calls", []), ensure_ascii=False)) for message in messages)
            if size > run.input_limit:
                raise BudgetExceeded("input_characters")
            run.model_calls += 1
            run.event("model_started", call=run.model_calls)
            reply = await asyncio.wait_for(self.model.ainvoke(messages), timeout=run.remaining())
            if not isinstance(reply, AIMessage):
                raise ValueError("Invalid model response")
            for key in run.usage:
                value = (reply.usage_metadata or {}).get(key, 0)
                if isinstance(value, int) and value >= 0:
                    run.usage[key] += value
            run.event("model_completed", call=run.model_calls)
            if getattr(reply, "invalid_tool_calls", []):
                raise ValueError("Invalid tool arguments")
            messages.append(reply)
            calls = reply.tool_calls or []
            if not calls:
                if not isinstance(reply.content, str) or not reply.content.strip():
                    raise ValueError("Empty model output")
                if len(reply.content) > run.output_limit:
                    raise BudgetExceeded("output_characters")
                run.event("completed")
                return {"messages": messages, "harness_report": run.report("completed")}
            # Validate the entire proposed batch before any tool is run.
            if run.tool_calls + len(calls) > run.tool_limit:
                raise BudgetExceeded("tool_calls")
            ids = [call.get("id") for call in calls]
            if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)) or executed_call_ids.intersection(ids):
                raise ValueError("Duplicate or missing tool call ID")
            for call in calls:
                if call.get("name") not in self.tools:
                    raise ValueError("Tool is outside selected capabilities")
                self.tools[call['name']].get_input_schema().model_validate(call.get('args'))
            for call in calls:
                run.remaining()
                name = call['name']
                run.tool_calls += 1
                executed_call_ids.add(call['id'])
                run.event("tool_started", tool=name, call=run.tool_calls)
                result = await asyncio.wait_for(self.tools[name].ainvoke(call['args']),
                                                timeout=min(run.tool_timeout, run.remaining()))
                content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
                if len(content) > run.output_limit:
                    raise BudgetExceeded("tool_output_characters")
                try:
                    parsed = json.loads(content)
                    failed = isinstance(parsed, dict) and parsed.get('ok') is False
                except (ValueError, TypeError):
                    failed = False
                run.event("tool_failed" if failed else "tool_completed", tool=name,
                          result_sha256=hashlib.sha256(content.encode()).hexdigest(), result_characters=len(content))
                messages.append(ToolMessage(content=content, tool_call_id=call['id'], name=name))
