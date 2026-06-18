#!/usr/bin/env python3
import time
import struct
import threading

try:
    import serial
    from serial import SerialException
except Exception as exc:
    serial = None
    SerialException = Exception
    _SERIAL_IMPORT_ERROR = exc
else:
    _SERIAL_IMPORT_ERROR = None


PRESET = {"code": 16, "address": 1000}
READ = {"code": 4, "address": 2000}


class GripperHandler:
    def __init__(self, port="/tmp/ttyUR", baudrate=115200, timeout=1.0, slaveID=9):
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.slaveID = int(slaveID)
        self._lock = threading.Lock()
        self.serialModbus = None
        self._connect()

    def _connect(self):
        if serial is None:
            raise ImportError(
                f"pyserial import failed: {_SERIAL_IMPORT_ERROR}. "
                "Install pyserial and remove wrong 'serial' module."
            )

        retries = 0
        last_exc = None
        while retries < 3:
            try:
                self.serialModbus = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                )
                break
            except SerialException as exc:
                last_exc = exc
                print(f"Failed to connect to serial port: {exc}")
                retries += 1
                time.sleep(1.0)

        if self.serialModbus is None or not self.serialModbus.is_open:
            raise RuntimeError(
                f"Serial port is not opened ({self.port}). Last error: {last_exc}"
            )

    def ensure_connected(self):
        if self.serialModbus is None or not self.serialModbus.is_open:
            raise RuntimeError(f"Serial port is not open ({self.port})")

    def deactivate(self):
        data = self.make_byte_data(PRESET, 3, [0, 0, 0, 0, 0, 0])
        return self.request_and_response(data)

    def activate(self):
        data = self.make_byte_data(PRESET, 3, [1, 0, 0, 0, 0, 0])
        return self.request_and_response(data)

    def read_status(self, registerNum=1):
        data = self.make_byte_data(READ, registerNum, [0, 0, 0, 0, 0, 0])
        return self.request_and_response(data)

    def set_gripper_value(self, position, speed, force):
        position = int(max(0, min(255, position)))
        speed = int(max(0, min(255, speed)))
        force = int(max(0, min(255, force)))
        data = self.make_byte_data(PRESET, 3, [9, 0, 0, position, speed, force])
        return self.request_and_response(data)

    def open(self, speed=255, force=120, position=0):
        return self.set_gripper_value(position=position, speed=speed, force=force)

    def close(self, speed=255, force=120, position=255):
        return self.set_gripper_value(position=position, speed=speed, force=force)

    def close_serial(self):
        try:
            if self.serialModbus is not None and self.serialModbus.is_open:
                self.serialModbus.close()
        except Exception:
            pass

    def request_and_response(self, data, expected_bytes=8):
        with self._lock:
            self.ensure_connected()
            self.serialModbus.reset_input_buffer()
            self.serialModbus.reset_output_buffer()

            crc = self.encode_crc16(data)
            request = data + struct.pack("<H", crc)

            print(f"[Gripper] TX ({len(request)} bytes): {request.hex()}")
            self.serialModbus.write(request)

            response = self.serialModbus.read(expected_bytes)
            print(f"[Gripper] RX ({len(response)} bytes): {response.hex() if response else '<empty>'}")

            if not response:
                raise RuntimeError(
                    f"No response from gripper on {self.port}. "
                    f"Check device path, power, slaveID={self.slaveID}, baudrate={self.baudrate}, "
                    f"and whether another process is using the port."
                )

            return response

    def make_byte_data(self, register, registerNum, values):
        code = register["code"]
        address = register["address"]

        if code == 16:
            if len(values) != registerNum * 2:
                raise ValueError(
                    f"Expected {registerNum * 2} values, got {len(values)}"
                )
            data = (
                self.to_byte(self.slaveID) +
                self.to_byte(code) +
                self.to_byte(address, 2) +
                self.to_byte(registerNum, 2) +
                self.to_byte(registerNum * 2)
            )
            for v in values:
                data += self.to_byte(v)

        elif code == 4:
            data = (
                self.to_byte(self.slaveID) +
                self.to_byte(code) +
                self.to_byte(address, 2) +
                self.to_byte(registerNum, 2)
            )
        else:
            raise ValueError(f"Unsupported register code: {code}")

        return data

    def to_byte(self, value: int, length: int = 1) -> bytes:
        if length == 1:
            return struct.pack(">B", int(value))
        if length == 2:
            return struct.pack(">H", int(value))
        if length == 4:
            return struct.pack(">L", int(value))
        raise ValueError(f"Not supported byte length: {length}")

    def encode_crc16(self, data):
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc