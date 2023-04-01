"""Implements abstract classes that control agent behavior"""
from __future__ import annotations  # types will be strings by default in 3.11

from typing import TYPE_CHECKING
import logging

import numpy as np

from elfpy.main import SimulationState, Wallet, MarketAction, MarketActionType, Quantity, TokenType, Position

if TYPE_CHECKING:
    from typing import Optional, Iterable
