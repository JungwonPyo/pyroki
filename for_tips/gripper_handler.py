import serial
import time
import struct

# register info
PRESET = {
    'code' : 16,
    'address' : 1000
}
READ = {
    'code' : 4,
    'address' : 2000
}

class GripperHandler:
    def __init__(self, port = '/tmp/ttyUR', baudrate=115200, timeout=1, slaveID = 9):
        self.slaveID = slaveID
        # connect to serial port
        retries = 0
        while retries < 5:
            try:
                self.serialModbus = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
                break
            except serial.SerialException as e:
                print(f"Failed to connect to serial port: {e}")
            retries += 1
            time.sleep(1)
        assert self.serialModbus.is_open, f"Serial port is not opened ({port})"

    # commands
    def deactivate(self):
        data = self.makeByteData(PRESET, 3, [0,0,0,0,0,0])
        response = self.request_and_response(data)
        return response

    def activate(self):
        data = self.makeByteData(PRESET, 3, [1,0,0,0,0,0])
        response = self.request_and_response(data)
        return response

    def read_status(self, registerNum = 1):
        data = self.makeByteData(READ, registerNum, [0,0,0,0,0,0])
        response = self.request_and_response(data)
        return response

    def set_gripper_value(self, position, speed, force) -> bytes: # pos 0이 open, 255가 close, force 0이면 전류제어
        data = self.makeByteData(PRESET, 3, [9, 0, 0, position, speed, force])
        response = self.request_and_response(data)
        return response

    def request_and_response(self, data):
        self.serialModbus.reset_input_buffer()
        self.serialModbus.reset_output_buffer()
        crc = self.encode_crc16(data)
        request = data + struct.pack('<H', crc)
        print(f"Sending request: {request.hex()}")
        self.serialModbus.write(request)
        response = self.serialModbus.read(8)
        print(f"Received response: {response.hex()}")
        if not response:
            print("Not responsed from gripper.")
        return response

    # utils
    def makeByteData(self, register, registerNum, values):
        code = register['code']
        address = register['address']
        if code == 16:
            assert len(values) == registerNum*2, "Using preset registers, not match number of bytes with of values"
            data = self.toByte(self.slaveID) + self.toByte(code) + self.toByte(address, 2) + self.toByte(registerNum, 2) + self.toByte(registerNum*2)
            for v in values:
                data += self.toByte(v)
        elif code == 4:
            data = self.toByte(self.slaveID) + self.toByte(code) + self.toByte(address, 2) + self.toByte(registerNum, 2)
        else:
            raise Exception("Not available register code. only use preset registers and input registers")
        return data

    def toByte(self, value : int, length : int = 1) -> bytes:
        if length == 1:
            return struct.pack('>B', value)
        elif length == 2:
            return struct.pack('>H', value)
        elif length == 4:
            return struct.pack('>L', value)
        else:
            raise Exception(f"Not supported byte length : {value}")

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
