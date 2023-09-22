# pylint: disable=wildcard-import,unused-import,missing-class-docstring,missing-module-docstring,missing-final-newline
from dataclasses import dataclass

from .random_agent import RandomAgent
from .smart_long import LongLouie
from .smart_short import ShortSally
from .deterministic import DBot
from .minimal import MBot
from .oneline import OBot
from .sheng import ShengTsung

@dataclass
class Policies:
    random_agent = RandomAgent
    long_louie = LongLouie
    short_sally = ShortSally
    deterministic = DBot
    minimal = MBot
    oneline = OBot
    ShengTsung = ShengTsung
