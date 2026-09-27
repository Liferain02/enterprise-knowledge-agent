"""Explicit units for systems research; bit/byte and SI/IEC stay distinct."""
from src.agent.skills._validation import finite, json_result

# dimension, multiplier to seconds / bytes / bytes per second / hertz
UNITS = {
    "ns": ("time", 1e-9), "us": ("time", 1e-6), "µs": ("time", 1e-6),
    "ms": ("time", 1e-3), "s": ("time", 1), "min": ("time", 60), "h": ("time", 3600),
    "bit": ("data", 0.125), "B": ("data", 1), "KB": ("data", 1000),
    "MB": ("data", 1e6), "GB": ("data", 1e9), "TB": ("data", 1e12),
    "KiB": ("data", 1024), "MiB": ("data", 1024**2), "GiB": ("data", 1024**3),
    "TiB": ("data", 1024**4), "bps": ("rate", 0.125), "Kbps": ("rate", 125),
    "Mbps": ("rate", 125000), "Gbps": ("rate", 125000000),
    "B/s": ("rate", 1), "KB/s": ("rate", 1000), "MB/s": ("rate", 1e6),
    "GB/s": ("rate", 1e9), "KiB/s": ("rate", 1024), "MiB/s": ("rate", 1024**2),
    "GiB/s": ("rate", 1024**3), "Hz": ("frequency", 1),
    "kHz": ("frequency", 1e3), "MHz": ("frequency", 1e6), "GHz": ("frequency", 1e9),
}


def _unit(name: str):
    if name not in UNITS:
        raise ValueError("未知或大小写不正确的单位；支持 ns/us/ms/s/min/h、bit/B/KB/MB/GB/TB、KiB/MiB/GiB/TiB、bps/Kbps/Mbps/Gbps、B/s/KB/s/MB/s/GB/s/KiB/s/MiB/s/GiB/s、Hz/kHz/MHz/GHz")
    return UNITS[name]


@json_result
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """换算时间、数据大小、带宽或频率。单位区分大小写：MB 是十进制字节，MiB 是二进制字节，Mbps 是比特率。"""
    source, target = _unit(from_unit), _unit(to_unit)
    if source[0] != target[0]:
        raise ValueError("不能换算不同物理维度")
    return {"input": finite(value), "from_unit": from_unit, "to_unit": to_unit,
            "value": finite(value) * source[1] / target[1]}


@json_result
def estimate_transfer_time(size: float, size_unit: str, bandwidth: float, bandwidth_unit: str,
                           efficiency: float = 1.0) -> str:
    """估算传输秒数：数据量/有效带宽。效率由用户给定，范围 (0,1]；这是理论估算，不是实测。"""
    data, rate = _unit(size_unit), _unit(bandwidth_unit)
    size, bandwidth, efficiency = finite(size), finite(bandwidth), finite(efficiency)
    if data[0] != "data" or rate[0] != "rate":
        raise ValueError("size_unit 必须是数据量，bandwidth_unit 必须是传输速率")
    if size < 0 or bandwidth <= 0 or not 0 < efficiency <= 1:
        raise ValueError("大小必须非负，带宽必须为正，效率必须在 (0,1]")
    return {"seconds": size * data[1] / (bandwidth * rate[1] * efficiency),
            "efficiency": efficiency, "assumption": "固定有效带宽，未单独建模握手、排队、磁盘和协议开销"}
