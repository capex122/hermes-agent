#!/usr/bin/env python3
"""Insert new anti-detection fingerprint tests into test_browser_hardening.py"""

# New tests to add
new_tests = '''    def test_fingerprint_pool_uses_dict_format_with_extended_attributes(self):
        """Verify pool entries include Canvas, Audio, Font, hardware attributes."""
        import tools.browser_tool as bt
        
        for profile in bt._FINGERPRINT_POOL:
            assert isinstance(profile, dict)
            assert "ua" in profile
            assert "platform" in profile
            assert "vendor" in profile
            assert "screen_w" in profile and "screen_h" in profile
            assert "timezone" in profile
            assert "hardware_concurrency" in profile
            assert "device_memory" in profile
            assert "max_touch_points" in profile
            assert "color_depth" in profile

    def test_validate_fingerprint_consistency_accepts_valid_profiles(self):
        """Consistency validator must accept real pool profiles."""
        import tools.browser_tool as bt
        
        for profile in bt._FINGERPRINT_POOL:
            is_valid = bt._validate_fingerprint_consistency(profile)
            assert is_valid, f"Profile {profile} failed validation"

    def test_validate_fingerprint_consistency_rejects_impossible_combinations(self):
        """Consistency validator must reject conflicting platform/UA pairs."""
        import tools.browser_tool as bt
        
        invalid_profile = {
            "ua": "Mozilla/5.0 (Windows NT 10.0...) Chrome/124",
            "platform": "MacIntel",
            "vendor": "Google Inc.",
            "screen_w": 1920,
            "screen_h": 1080,
            "timezone": "UTC",
            "hardware_concurrency": 8,
            "device_memory": 16,
            "max_touch_points": 0,
            "color_depth": 24,
        }
        assert not bt._validate_fingerprint_consistency(invalid_profile)

    def test_build_random_stealth_js_includes_canvas_spoofing(self):
        """Stealth JS must include Canvas fingerprint spoofing with noise injection."""
        import tools.browser_tool as bt
        
        js = bt._build_random_stealth_js(0)
        assert "toDataURL" in js
        assert "Canvas" in js or "canvas" in js
        assert "Math.random()" in js

    def test_build_random_stealth_js_includes_audio_spoofing(self):
        """Stealth JS must include AudioContext spoofing for audio fingerprint blocking."""
        import tools.browser_tool as bt
        
        js = bt._build_random_stealth_js(0)
        assert "AudioContext" in js
        assert "getChannelData" in js or "getByteFrequencyData" in js

    def test_build_random_stealth_js_includes_font_blocking(self):
        """Stealth JS must block font detection probes."""
        import tools.browser_tool as bt
        
        js = bt._build_random_stealth_js(0)
        assert "measureText" in js
        assert "font" in js.lower()

    def test_build_random_stealth_js_includes_webgl2_support(self):
        """Stealth JS must spoof both WebGL and WebGL2 renderers."""
        import tools.browser_tool as bt
        
        js = bt._build_random_stealth_js(0)
        assert "WebGL2RenderingContext" in js

    def test_fingerprint_profile_from_seed_returns_valid_profile(self):
        """Profile extraction must always return consistent valid profiles."""
        import tools.browser_tool as bt
        
        for seed in range(0, len(bt._FINGERPRINT_POOL) * 3):
            profile = bt._fingerprint_profile_from_seed(seed)
            assert isinstance(profile, dict)
            assert bt._validate_fingerprint_consistency(profile)

'''

# Read the test file
with open('tests/tools/test_browser_hardening.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find insertion point (before class TestProxyReputation:)
insertion_point = content.find('\nclass TestProxyReputation:')
if insertion_point == -1:
    print("ERROR: Could not find class TestProxyReputation:")
    exit(1)

# Insert the new tests
new_content = content[:insertion_point] + '\n' + new_tests + content[insertion_point:]

# Write back
with open('tests/tools/test_browser_hardening.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✓ Added 9 new comprehensive fingerprinting tests")
