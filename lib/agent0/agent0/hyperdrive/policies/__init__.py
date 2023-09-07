"""Generic bot imports used by every policy, abbreviated to be shorter."""
from fixedpointmath import FixedPoint
from numpy.random._generator import Generator as Rng

from agent0.base.policies import BasePolicy
from agent0.hyperdrive.state import HyperdriveMarketAction as Action
from agent0.hyperdrive.state import HyperdriveActionType as Type
from agent0.hyperdrive.agents import HyperdriveWallet
from elfpy.markets.hyperdrive import HyperdriveMarket as HyperdriveMarketState
from elfpy.types import MarketType, Trade

HYPERDRIVE = MarketType.HYPERDRIVE
fp100 = FixedPoint(100)
fp0 = FixedPoint(0)
