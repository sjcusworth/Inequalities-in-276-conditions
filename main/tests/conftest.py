"""
pytest configuration for the IncPrevMethods test suite.
Ensures the ANALOGY_SCIENTIFIC methods directory is on sys.path before
any test module is imported.
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_METHODS_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "ANALOGY_SCIENTIFIC",
    "analogy", "study_design", "incidence_prevalence"))

if _METHODS_DIR not in sys.path:
    sys.path.insert(0, _METHODS_DIR)
