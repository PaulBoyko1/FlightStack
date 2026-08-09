"""Compact binary HIL packet framing with CRC-16/CCITT."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from numbers import Integral

SYNC = b"FS"
_HEADER = struct.Struct("<2sBH")
_CRC = struct.Struct("<H")


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if (
        isinstance(initial, bool)
        or not isinstance(initial, Integral)
        or not 0 <= initial <= 0xFFFF
    ):
        raise ValueError("initial must fit in uint16")
    crc = int(initial)
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class Packet:
    message_id: int
    payload: bytes

    def __post_init__(self) -> None:
        if (
            isinstance(self.message_id, bool)
            or not isinstance(self.message_id, Integral)
            or not 0 <= self.message_id <= 255
        ):
            raise ValueError("message_id must fit in uint8")
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes")
        if len(self.payload) > 65535:
            raise ValueError("payload too large")

    def encode(self) -> bytes:
        body = _HEADER.pack(SYNC, int(self.message_id), len(self.payload)) + self.payload
        return body + _CRC.pack(crc16_ccitt(body))

    @classmethod
    def decode(cls, frame: bytes) -> Packet:
        if not isinstance(frame, bytes):
            raise TypeError("frame must be bytes")
        if len(frame) < _HEADER.size + _CRC.size:
            raise ValueError("frame too short")
        sync, message_id, payload_len = _HEADER.unpack_from(frame)
        if sync != SYNC:
            raise ValueError("invalid sync")
        expected_len = _HEADER.size + payload_len + _CRC.size
        if len(frame) != expected_len:
            raise ValueError("frame length mismatch")
        expected_crc = _CRC.unpack_from(frame, expected_len - _CRC.size)[0]
        if crc16_ccitt(frame[:-_CRC.size]) != expected_crc:
            raise ValueError("CRC mismatch")
        payload = frame[_HEADER.size : -_CRC.size]
        return cls(message_id, payload)
