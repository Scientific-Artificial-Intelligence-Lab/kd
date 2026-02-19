"""Root conftest: pre-initialize Julia/juliacall to prevent signal crashes.

juliacall must be initialized before any kd submodule is imported;
otherwise Julia's signal handlers conflict with Python's and cause
an abort (signal 6) on macOS.
"""
try:
    import juliacall
    juliacall.Main  # trigger lazy init
except Exception:
    pass
