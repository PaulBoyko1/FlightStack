import pytest
from flightstack.hil.protocol import Packet, crc16_ccitt


def test_crc_known_vector() -> None:
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_packet_round_trip() -> None:
    packet = Packet(7, b"hello\x00flight")
    assert Packet.decode(packet.encode()) == packet


def test_packet_detects_corruption() -> None:
    frame = bytearray(Packet(1, b"abc").encode())
    frame[5] ^= 0x01
    with pytest.raises(ValueError, match="CRC"):
        Packet.decode(bytes(frame))
