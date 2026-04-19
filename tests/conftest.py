"""
tests/conftest.py
==================
Shared fixtures available to all test files.

What is conftest.py?
  A special pytest file. Any fixture defined here is automatically
  available to every test file in the same directory — no imports
  needed. pytest discovers it by convention.

  This is where you put fixtures that multiple test files share.
  Right now it's minimal, but as the project grows you'd add things
  like database connections, auth tokens, or shared mock objects.
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path so imports work
# when running 'pytest' from the project root directory
sys.path.insert(0, str(Path(__file__).parent.parent))
