"""Policies for expert system trading bots."""
from __future__ import annotations

from typing import NamedTuple

from agent0.hyperdrive.policies.arbitrage import Arbitrage
from agent0.hyperdrive.policies.lpandarb import LPandArb
from agent0.hyperdrive.policies.opener import Opener
from agent0.hyperdrive.policies.random import Random
from agent0.hyperdrive.policies.smart_long import SmartLong
from agent0.hyperdrive.policies.deterministic import DBot
from agent0.hyperdrive.policies.minimal import MBot
from agent0.hyperdrive.policies.oneline import OBot as OneLineBot


# Container for all the policies
class Zoo(NamedTuple):
    """All policies in agent0."""

    random = Random
    arbitrage = Arbitrage
    smart_long = SmartLong
    lp_and_arb = LPandArb
    deterministic = DBot
    minimal = MBot
    oneline = OneLineBot
    opener = Opener

    def describe(self, policies: list | str | None = None) -> str:
        """Describe policies, either specific ones provided, or all of them."""
        # programmatically create a list with all the policies
        existing_policies = [
            attr for attr in dir(self) if not attr.startswith("_") and attr not in ["describe", "count", "index"]
        ]
        if policies is None:  # we are not provided specific policies to describe
            policies = existing_policies
        elif not isinstance(policies, list):  # not a list
            policies = [policies]  # we make it a list

        for policy in policies:
            if policy not in existing_policies:
                raise ValueError(f"Unknown policy: {policy}")

        return "\n".join([f"=== {policy} ===\n{getattr(self, policy).description()}" for policy in policies])
