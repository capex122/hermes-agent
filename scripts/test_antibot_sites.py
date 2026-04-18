#!/usr/bin/env python3
"""Run anti-bot smoke tests against a list of real-world detection pages.

This script exercises Hermes browser tooling the same way the agent does for
"search <url>" requests: it calls browser_search(url, task_id=...).

It records whether navigation succeeded, whether bot detection was flagged,
and whether blocked-page content was correctly marked as unusable.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.browser_tool import (
    browser_navigate,
    browser_search,
    check_browser_requirements,
    cleanup_browser,
)

DEFAULT_SITES = [
    "https://bot.incolumitas.com",
    "https://pixelscan.net/bot-check",
    "https://bot.sannysoft.com",
    "https://www.browserscan.net/bot-detection",
    "https://pixelscan.net",
    "https://demo.fingerprint.com/playground",
    "https://www.ipqualityscore.com/bot-management/bot-detection-check",
    "https://botguard.net/en/tools",
]


@dataclass
class SiteResult:
    site: str
    mode: str
    elapsed_seconds: float
    success: bool
    bot_detection_detected: bool
    challenge_pattern: str
    fresh_session_retry_attempted: bool
    fresh_session_retry_count: int
    challenge_wait_seconds: float
    blocked_page_content_available: bool
    content_from_blocked_page_must_not_be_used: bool
    error: str
    required_next_action: str
    final_url: str
    title: str


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"success": False, "error": "Non-dict JSON response"}
    except Exception as exc:
        return {"success": False, "error": f"Invalid JSON response: {exc}", "raw": raw[:500]}


def _run_site(site: str, mode: str, task_id: str) -> SiteResult:
    start = time.perf_counter()
    if mode == "navigate":
        payload = _safe_json(browser_navigate(site, task_id=task_id))
    else:
        payload = _safe_json(browser_search(site, task_id=task_id))
    elapsed = time.perf_counter() - start

    return SiteResult(
        site=site,
        mode=mode,
        elapsed_seconds=round(elapsed, 2),
        success=bool(payload.get("success")),
        bot_detection_detected=bool(payload.get("bot_detection_detected")),
        challenge_pattern=str(payload.get("challenge_pattern") or ""),
        fresh_session_retry_attempted=bool(payload.get("fresh_session_retry_attempted")),
        fresh_session_retry_count=int(payload.get("fresh_session_retry_count") or 0),
        challenge_wait_seconds=float(payload.get("challenge_wait_seconds") or 0.0),
        blocked_page_content_available=bool(payload.get("blocked_page_content_available", payload.get("success", False))),
        content_from_blocked_page_must_not_be_used=bool(payload.get("content_from_blocked_page_must_not_be_used")),
        error=str(payload.get("error") or ""),
        required_next_action=str(payload.get("required_next_action") or ""),
        final_url=str(payload.get("url") or ""),
        title=str(payload.get("title") or ""),
    )


def _print_summary(results: list[SiteResult]) -> None:
    print("\nAnti-bot test summary")
    print("=" * 100)
    print(f"{'Site':48} {'Status':9} {'Bot?':5} {'Retry':5} {'Guarded':7} {'Seconds':7}")
    print("-" * 100)

    for r in results:
        status = "OK" if r.success else "FAIL"
        bot = "yes" if r.bot_detection_detected else "no"
        retry = str(r.fresh_session_retry_count)
        guarded = "yes" if r.content_from_blocked_page_must_not_be_used else "no"
        site = (r.site[:45] + "...") if len(r.site) > 48 else r.site
        print(f"{site:48} {status:9} {bot:5} {retry:5} {guarded:7} {r.elapsed_seconds:7.2f}")

    ok_count = sum(1 for r in results if r.success)
    bot_count = sum(1 for r in results if r.bot_detection_detected)
    guarded_count = sum(1 for r in results if r.content_from_blocked_page_must_not_be_used)
    print("-" * 100)
    print(
        f"Total={len(results)} | success={ok_count} | bot_detection={bot_count} | "
        f"blocked_content_guarded={guarded_count}"
    )

    print("\nPer-site details")
    print("=" * 100)
    for r in results:
        print(f"Site: {r.site}")
        print(f"  status: {'success' if r.success else 'failure'}")
        print(f"  bot_detection_detected: {r.bot_detection_detected}")
        if r.challenge_pattern:
            print(f"  challenge_pattern: {r.challenge_pattern}")
        print(f"  fresh_session_retry_count: {r.fresh_session_retry_count}")
        if r.challenge_wait_seconds:
            print(f"  challenge_wait_seconds: {r.challenge_wait_seconds}")
        print(f"  content_from_blocked_page_must_not_be_used: {r.content_from_blocked_page_must_not_be_used}")
        if r.error:
            err = r.error if len(r.error) <= 220 else r.error[:217] + "..."
            print(f"  error: {err}")
        if r.required_next_action:
            action = r.required_next_action if len(r.required_next_action) <= 220 else r.required_next_action[:217] + "..."
            print(f"  required_next_action: {action}")
        if r.final_url:
            print(f"  final_url: {r.final_url}")
        if r.title:
            print(f"  title: {r.title}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run anti-bot smoke tests against known test pages")
    parser.add_argument("--mode", choices=["search", "navigate"], default="search")
    parser.add_argument("--site", action="append", default=[], help="Site URL to test (repeatable)")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write JSON results",
    )
    args = parser.parse_args()

    if not check_browser_requirements():
        print("Browser requirements are not met. Install/configure agent-browser or cloud browser provider first.")
        return 2

    sites = args.site if args.site else DEFAULT_SITES
    results: list[SiteResult] = []

    for idx, site in enumerate(sites, start=1):
        task_id = f"antibot-test-{idx}"
        print(f"[{idx}/{len(sites)}] testing {site} using mode={args.mode} ...")
        try:
            result = _run_site(site, args.mode, task_id)
        except Exception as exc:
            result = SiteResult(
                site=site,
                mode=args.mode,
                elapsed_seconds=0.0,
                success=False,
                bot_detection_detected=False,
                challenge_pattern="",
                fresh_session_retry_attempted=False,
                fresh_session_retry_count=0,
                challenge_wait_seconds=0.0,
                blocked_page_content_available=False,
                content_from_blocked_page_must_not_be_used=False,
                error=f"script exception: {exc}",
                required_next_action="",
                final_url="",
                title="",
            )
        finally:
            try:
                cleanup_browser(task_id)
            except Exception:
                pass

        results.append(result)

    _print_summary(results)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON report written to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
