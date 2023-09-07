# pylint: disable=wildcard-import,unused-import,missing-class-docstring,missing-module-docstring,missing-final-newline
from dataclasses import dataclass

from .random_agent import RandomAgent
from .deterministic import DBot
from .minimal import MBot
from .oneline import OBot
from .arbitrage import ArbitragePolicy

@dataclass
class Policies:
    random_agent = RandomAgent
    arbitrage = ArbitragePolicy
    deterministic = DBot
    minimal = MBot
    oneline = OBot