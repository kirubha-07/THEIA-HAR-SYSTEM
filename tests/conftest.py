"""
conftest.py — Pytest configuration and custom command-line options.
"""

def pytest_addoption(parser):
    parser.addoption(
        "--clip",
        action="store",
        default=None,
        help="Path or filename of video clip for parity / evidence tests.",
    )
