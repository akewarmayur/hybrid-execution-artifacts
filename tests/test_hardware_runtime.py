"""Focused tests for live-hardware runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass

from checkrcq_eval.common.hardware_runtime import _counts_for_pub_result, get_runtime_backend


class _CountsBox:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


@dataclass
class _DataBox:
    meas: _CountsBox

    def keys(self) -> list[str]:
        return ["meas"]

    def __getitem__(self, key: str) -> _CountsBox:
        if key != "meas":
            raise KeyError(key)
        return self.meas


@dataclass
class _PubResult:
    data: _DataBox


class _Service:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def backend(self, backend_name: str, instance: str | None = None) -> tuple[str, str | None]:
        self.calls.append((backend_name, instance))
        return (backend_name, instance)


def test_counts_for_pub_result_reads_meas_register() -> None:
    pub_result = _PubResult(data=_DataBox(meas=_CountsBox({"00": 5, "11": 3})))
    assert _counts_for_pub_result(pub_result) == {"00": 5, "11": 3}


def test_get_runtime_backend_passes_instance_when_available() -> None:
    service = _Service()
    backend = get_runtime_backend(service, "ibm_pittsburgh", instance="crn:test")
    assert backend == ("ibm_pittsburgh", "crn:test")
    assert service.calls == [("ibm_pittsburgh", "crn:test")]
