"""用真实浏览器运行最小多轮科研问答验收集。

该脚本只复用冻结数据集中的 8 条代表样本，不生成新问题、不修改知识库。
每条样本创建独立会话，完成后立即删除。建议在关闭 Mem0 的独立测试实例运行。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
from pathlib import Path

from playwright.async_api import Page, async_playwright

from config.settings import get_settings
from tests.eval.multi_turn_coreference_dataset import MULTI_TURN_COREFERENCE_CASES


SELECTED_CASE_IDS = (
    "mt-pronoun-01",
    "mt-pronoun-03",
    "mt-followup-01",
    "mt-followup-04",
    "mt-compare-01",
    "mt-compare-02",
    "mt-explicit-02",
    "mt-explicit-08",
)
BASE_URL = os.getenv("EKA_TEST_BASE_URL", "http://127.0.0.1:8011").rstrip("/")
OUTPUT_PATH = Path("data/多轮真实浏览器验收结果.json")
CHROMIUM_PATH = os.getenv(
    "PLAYWRIGHT_CHROMIUM_PATH",
    "/share/home/lifr/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome",
)


async def _ask(page: Page, question: str, timeout_ms: int = 120_000) -> dict:
    before = await page.locator(".message.assistant .message-body").count()
    box = page.locator('textarea[placeholder="输入实验室问题，按 Enter 发送..."]')
    await box.fill(question)
    started = time.monotonic()
    await box.press("Enter")
    await page.wait_for_function(
        """before => {
          const answers = [...document.querySelectorAll('.message.assistant .message-body')]
            .filter(node => !node.textContent.includes('正在思考中'));
          const send = document.querySelector('.send-btn');
          return answers.length > before
            && !send?.classList.contains('is-loading')
            && (answers.at(-1)?.textContent || '').trim().length > 0;
        }""",
        arg=before,
        timeout=timeout_ms,
    )
    message = page.locator(".message.assistant").last
    return {
        "latency_seconds": round(time.monotonic() - started, 2),
        "answer_length": len((await message.locator(".message-body").inner_text()).strip()),
        "agent_badge": (await message.locator(".agent-badge").inner_text()).strip(),
        "source_card_count": await message.locator(".source-card").count(),
    }


def _source_matches(source: dict, expected: str) -> bool:
    candidates = {
        str(source.get("source") or ""),
        str(source.get("title") or ""),
    }
    expected_stem = Path(expected).stem
    return any(expected in value or expected_stem in value for value in candidates)


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 2)


def _build_result(
    *,
    cases: list[dict],
    bad_responses: list[dict],
    console_errors: list[str],
    started: float,
    status: str,
) -> dict:
    completed = sum(bool(case.get("completed")) for case in cases)
    route_passed = sum(bool(case.get("route_ok")) for case in cases)
    source_passed = sum(bool(case.get("gold_hit")) for case in cases)
    latencies = [
        float(turn["latency_seconds"])
        for case in cases
        for turn in (case.get("first_turn"), case.get("followup_turn"))
        if isinstance(turn, dict) and isinstance(turn.get("latency_seconds"), (int, float))
    ]
    return {
        "status": status,
        "dataset": "lab-multi-turn-coreference-v1",
        "selected_case_ids": list(SELECTED_CASE_IDS),
        "base_url": BASE_URL,
        "mem0_expected_disabled": True,
        "summary": {
            "total": len(SELECTED_CASE_IDS),
            "attempted": len(cases),
            "completed": completed,
            "route_passed": route_passed,
            "gold_source_hit": source_passed,
            "latency_sample_count": len(latencies),
            "latency_p50_seconds": round(statistics.median(latencies), 2) if latencies else None,
            "latency_p95_seconds": _percentile_95(latencies),
            "http_error_count": len(bad_responses),
            "console_error_count": len(console_errors),
            "elapsed_seconds": round(time.monotonic() - started, 2),
        },
        "bad_responses": bad_responses,
        "console_errors": console_errors,
        "cases": cases,
    }


def _save_checkpoint(
    *,
    cases: list[dict],
    bad_responses: list[dict],
    console_errors: list[str],
    started: float,
    status: str,
) -> dict:
    result = _build_result(
        cases=cases,
        bad_responses=bad_responses,
        console_errors=console_errors,
        started=started,
        status=status,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


async def main() -> None:
    settings = get_settings()
    selected = {
        case.case_id: case for case in MULTI_TURN_COREFERENCE_CASES
        if case.case_id in SELECTED_CASE_IDS
    }
    if set(selected) != set(SELECTED_CASE_IDS):
        raise RuntimeError("验收样本编号与冻结数据集不一致")

    cases = []
    bad_responses: list[dict] = []
    console_errors: list[str] = []
    started = time.monotonic()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=CHROMIUM_PATH,
            headless=True,
        )
        context = await browser.new_context(viewport={"width": 1440, "height": 1000})
        page = await context.new_page()
        page.on(
            "response",
            lambda response: bad_responses.append({
                "status": response.status,
                "url": response.url,
            }) if response.status >= 400 else None,
        )
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error" else None,
        )

        await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
        inputs = page.locator(".login-input")
        await inputs.nth(0).fill(settings.admin_username)
        await inputs.nth(1).fill(settings.admin_password)
        async with page.expect_response(
            lambda response: response.request.method == "GET"
            and response.url.endswith("/api/v1/history/default")
        ) as login_history_info:
            await page.locator(".login-btn").click()
        await login_history_info.value
        await page.locator(".user-info").wait_for(state="visible", timeout=30_000)
        token = await page.evaluate("localStorage.getItem('eka_token')")

        for case_id in SELECTED_CASE_IDS:
            case = selected[case_id]
            session_id = ""
            record = {
                "case_id": case.case_id,
                "category": case.category,
                "previous_user_query": case.previous_user_query,
                "followup_query": case.followup_query,
                "gold_sources": list(case.gold_sources),
            }
            try:
                async with (
                    page.expect_response(
                        lambda response: response.request.method == "POST"
                        and response.url.endswith("/api/v1/sessions")
                    ) as response_info,
                    page.expect_response(
                        lambda response: response.request.method == "GET"
                        and "/api/v1/history/session_" in response.url
                    ) as new_history_info,
                ):
                    await page.locator(".btn-new-session").click()
                session_id = (await (await response_info.value).json())["session_id"]
                await new_history_info.value
                await page.locator(".empty-state").wait_for(state="visible", timeout=30_000)
                await page.wait_for_function(
                    "() => document.querySelectorAll('.message').length === 0",
                    timeout=30_000,
                )

                record["first_turn"] = await _ask(page, case.previous_user_query)
                record["followup_turn"] = await _ask(page, case.followup_query)

                history_response = await context.request.get(
                    f"{BASE_URL}/api/v1/history/{session_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                history = await history_response.json()
                assistant_messages = [
                    message for message in history.get("messages", [])
                    if message.get("role") == "assistant"
                ]
                sources = (
                    assistant_messages[-1].get("metadata", {}).get("sources", [])
                    if assistant_messages else []
                )
                matched = [
                    expected for expected in case.gold_sources
                    if any(_source_matches(source, expected) for source in sources)
                ]
                record.update({
                    "history_status": history_response.status,
                    "followup_agent": (
                        assistant_messages[-1].get("metadata", {}).get("agent")
                        if assistant_messages else ""
                    ),
                    "source_titles": [source.get("title", "") for source in sources],
                    "matched_gold_sources": matched,
                    "route_ok": (
                        assistant_messages[-1].get("metadata", {}).get("agent")
                        == "knowledge_agent"
                    ) if assistant_messages else False,
                    "gold_hit": bool(matched),
                    "completed": True,
                })
            except Exception as exc:
                record.update({
                    "completed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc).splitlines()[0][:300],
                })
            finally:
                if session_id:
                    try:
                        await context.request.delete(
                            f"{BASE_URL}/api/v1/sessions/{session_id}",
                            headers={"Authorization": f"Bearer {token}"},
                        )
                    except Exception as cleanup_exc:
                        record["cleanup_error_type"] = type(cleanup_exc).__name__
            cases.append(record)
            _save_checkpoint(
                cases=cases,
                bad_responses=bad_responses,
                console_errors=console_errors,
                started=started,
                status="in_progress",
            )
            print(json.dumps({
                "case_id": case_id,
                "completed": record.get("completed", False),
                "route_ok": record.get("route_ok", False),
                "gold_hit": record.get("gold_hit", False),
            }, ensure_ascii=False), flush=True)

        await browser.close()

    result = _save_checkpoint(
        cases=cases,
        bad_responses=bad_responses,
        console_errors=console_errors,
        started=started,
        status="completed",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
