"""Generic bot imports used by every policy, abbreviated to be shorter."""
from fixedpointmath import FixedPoint
from numpy.random._generator import Generator as Rng

from agent0.base.policies import BasePolicy
from agent0.hyperdrive.state import HyperdriveMarketAction as Action
from agent0.hyperdrive.state import HyperdriveActionType as Type
from elfpy.markets.hyperdrive import HyperdriveMarket as HyperdriveMarketState
from elfpy.types import MarketType, Trade
from .hyperdrive_policy import HyperdrivePolicy

HYPERDRIVE = MarketType.HYPERDRIVE
fp100 = FixedPoint(100)
fp0 = FixedPoint(0)
