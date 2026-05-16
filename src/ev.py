"""
Compatibility module.

The EV/odds logic lives in utils.py to avoid duplicate implementations.
Keep this file so old imports from src.ev continue working.
"""

from src.utils import (
    american_profit_multiplier,
    american_to_probability,
    calculate_ev,
    calculate_profit,
)