"""
Global Configuration and Imports
"""
import os
import sys
from pathlib import Path
import yaml
from typing import List, Tuple, Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# App defaults used by modules that import `config` directly.
CLIPBOARD_MAX_ITEMS = 20
