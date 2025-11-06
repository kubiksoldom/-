import sys
from pathlib import Path
from typing import Callable, Iterable

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import api_guard  # noqa: E402


@pytest.fixture(autouse=True)
def patch_rate_controls(monkeypatch):
    class _DummyBucket:
        def take(self) -> bool:
            return True

    monkeypatch.setattr(api_guard, "_bucket", _DummyBucket())
    monkeypatch.setattr(api_guard, "_respect_min_delay", lambda: None)
    monkeypatch.setattr(api_guard, "_sleep_backoff", lambda *_, **__: None)
    monkeypatch.setattr(api_guard, "_bump_metric", lambda *_, **__: None)


def call_sequence(responses: Iterable[dict]) -> Callable:
    iterator = iter(responses)

    def _call(*args, **kwargs):
        try:
            return next(iterator)
        except StopIteration:  # pragma: no cover - safety
            return {"retCode": 0, "retMsg": "ok"}

    return _call


def test_safe_request_success_no_retry():
    fn = call_sequence(([{"retCode": 0, "retMsg": ""}]))
    resp = api_guard.safe_request(fn)
    assert resp["retCode"] == 0


def test_safe_request_accepts_leverage_not_modified():
    fn = call_sequence(([{"retCode": 110043, "retMsg": "unchanged"}]))
    resp = api_guard.safe_request(fn)
    assert resp["retCode"] == 110043


def test_safe_request_retries_timestamp_then_succeeds():
    fn = call_sequence((
        {"retCode": 10002, "retMsg": "timestamp"},
        {"retCode": 0, "retMsg": "ok"},
    ))
    resp = api_guard.safe_request(fn, max_tries=2)
    assert resp["retCode"] == 0


def test_safe_request_retries_rate_limit():
    fn = call_sequence((
        {"retCode": 10006, "retMsg": "rate limit"},
        {"retCode": 0, "retMsg": "ok"},
    ))
    resp = api_guard.safe_request(fn, max_tries=3)
    assert resp["retCode"] == 0


def test_safe_request_raises_on_params_error():
    fn = call_sequence(([{"retCode": 10001, "retMsg": "invalid"}]))
    with pytest.raises(RuntimeError):
        api_guard.safe_request(fn, max_tries=1)
