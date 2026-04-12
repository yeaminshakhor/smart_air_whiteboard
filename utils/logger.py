import os
import sys

try:
    import config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
import logging
from typing import Optional

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Placeholder for logger utility"""
    logging.basicConfig(level=level)
    return logging.getLogger(name)
