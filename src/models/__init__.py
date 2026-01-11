"""Models package."""

import sys
from pathlib import Path

# Add ImageBind to Python path to make it importable
_imagebind_path = Path(__file__).parent / "imagebind"
if str(_imagebind_path) not in sys.path:
    sys.path.insert(0, str(_imagebind_path))

# Make ImageBind available at package level
try:
    from imagebind import data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    
    __all__ = ['data', 'imagebind_model', 'ModalityType']
except ImportError:
    __all__ = []

