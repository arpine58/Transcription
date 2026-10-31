"""Print the OpenVINO devices on this machine. We want GPU (Intel Arc iGPU) and ideally NPU in the list."""
import openvino as ov

core = ov.Core()
for dev in core.available_devices:
    print(f"{dev:5} {core.get_property(dev, 'FULL_DEVICE_NAME')}")
