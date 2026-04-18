from unittest.mock import MagicMock, patch

import tools.webplus_service_manager as manager


def setup_function():
    manager._service_process = None
    manager._service_log_handle = None


def teardown_function():
    manager._service_process = None
    manager._service_log_handle = None


def test_bundled_web_service_is_healthy_true_when_healthz_ok():
    with patch.object(manager, "_load_cfg", return_value={"enabled": True, "base_url": "http://127.0.0.1:8765"}), patch.object(
        manager,
        "_request_json",
        return_value={"ok": True},
    ):
        assert manager.bundled_web_service_is_healthy() is True


def test_ensure_bundled_web_service_starts_process_when_needed():
    fake_process = MagicMock()
    fake_process.poll.side_effect = [None, None, None]

    with patch.object(
        manager,
        "_load_cfg",
        return_value={"enabled": True, "base_url": "http://127.0.0.1:8765", "auto_start": True},
    ), patch.object(manager, "bundled_web_service_is_healthy", side_effect=[False, False, True]), patch.object(
        manager,
        "_spawn_service_process",
        return_value=fake_process,
    ) as mock_spawn:
        assert manager.ensure_bundled_web_service() is True

    mock_spawn.assert_called_once_with("http://127.0.0.1:8765")


def test_shutdown_bundled_web_service_terminates_managed_process():
    fake_process = MagicMock()
    fake_process.poll.side_effect = [None, 0, 0]
    manager._service_process = fake_process

    with patch.object(
        manager,
        "_load_cfg",
        return_value={"enabled": True, "base_url": "http://127.0.0.1:8765"},
    ), patch.object(manager, "_request_json", return_value={"success": True}):
        manager.shutdown_bundled_web_service()

    fake_process.terminate.assert_not_called()