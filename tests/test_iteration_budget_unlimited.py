"""Tests for the IterationBudget unlimited override."""
import os
import threading

import pytest

from run_agent import IterationBudget


def test_default_caps_at_max():
    b = IterationBudget(3)
    assert b.consume() is True
    assert b.consume() is True
    assert b.consume() is True
    assert b.consume() is False
    assert b.remaining == 0


def test_unlimited_constructor_flag_bypasses_cap():
    b = IterationBudget(2, unlimited=True)
    for _ in range(50):
        assert b.consume() is True
    assert b.unlimited is True
    # remaining returns a huge sentinel
    assert b.remaining > 1_000_000


def test_unlimited_env_var_bypasses_cap(monkeypatch):
    monkeypatch.setenv("HERMES_UNLIMITED_ITERATIONS", "1")
    b = IterationBudget(2)
    for _ in range(20):
        assert b.consume() is True
    assert b.unlimited is True


def test_env_var_falsy_keeps_cap(monkeypatch):
    monkeypatch.setenv("HERMES_UNLIMITED_ITERATIONS", "0")
    b = IterationBudget(2)
    assert b.unlimited is False
    assert b.consume() is True
    assert b.consume() is True
    assert b.consume() is False


def test_set_unlimited_runtime_toggle():
    b = IterationBudget(1)
    assert b.consume() is True
    assert b.consume() is False  # cap reached

    b.set_unlimited(True)
    assert b.consume() is True
    assert b.consume() is True

    b.set_unlimited(False)
    # Now back to capped — used (3) >= max (1)
    assert b.consume() is False


def test_refund_works_in_unlimited_mode():
    b = IterationBudget(5, unlimited=True)
    b.consume()
    b.consume()
    assert b.used == 2
    b.refund()
    assert b.used == 1


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_env_var_truthy_values(monkeypatch, val):
    monkeypatch.setenv("HERMES_UNLIMITED_ITERATIONS", val)
    assert IterationBudget(1).unlimited is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "bogus"])
def test_env_var_falsy_values(monkeypatch, val):
    monkeypatch.setenv("HERMES_UNLIMITED_ITERATIONS", val)
    assert IterationBudget(1).unlimited is False
