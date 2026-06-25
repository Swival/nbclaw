"""nbclaw — No Bullshit Claw.

A 24/7 daemon that drives a swival agent from Signal: send it commands, get
answers, and schedule or cancel recurring tasks.
"""

from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
