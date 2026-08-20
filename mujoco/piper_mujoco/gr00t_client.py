"""Small numeric-only client for the official GR00T ZeroMQ policy server."""

from __future__ import annotations

from typing import Any

import msgpack
import msgpack_numpy
import zmq


class Gr00tClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 120_000):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(f"tcp://{host}:{port}")

    @staticmethod
    def _pack(value: Any) -> bytes:
        return msgpack.packb(value, default=msgpack_numpy.encode)

    @staticmethod
    def _unpack(value: bytes) -> Any:
        return msgpack.unpackb(value, object_hook=msgpack_numpy.decode, raw=False)

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self._socket.send(self._pack(request))
        response = self._unpack(self._socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def ping(self) -> bool:
        response = self.call("ping")
        return response.get("status") == "ok"

    def reset(self) -> None:
        self.call("reset", {"options": None})

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call(
            "get_action",
            {"observation": observation, "options": None},
        )
        return response[0], response[1]

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()

    def __enter__(self) -> "Gr00tClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
