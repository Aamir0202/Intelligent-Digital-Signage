import os
import sys


external_lib_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "external"))

if external_lib_path not in sys.path:
    sys.path.append(external_lib_path)
