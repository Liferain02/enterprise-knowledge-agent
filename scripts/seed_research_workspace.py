"""幂等初始化高性能计算实验室科研空间样例。"""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.services.research_service import research_service


if __name__ == "__main__":
    result = research_service.seed_lab_samples({"username": "lab-admin", "role": "admin"})
    print(result)
    print(research_service.get_overview({"username": "lab-admin", "role": "admin"}))
