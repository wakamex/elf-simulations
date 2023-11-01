"""Tests for hyperdrive/api.py"""
from __future__ import annotations

from typing import cast

from eth_typing import URI
from ethpy import EthConfig
from ethpy.base.transactions import smart_contract_read
from ethpy.hyperdrive.addresses import HyperdriveAddresses
from ethpy.hyperdrive.api import HyperdriveInterface
from ethpy.test_fixtures.local_chain import LocalHyperdriveChain
from fixedpointmath import FixedPoint
from web3 import HTTPProvider


class TestHyperdriveInterface:
    """Tests for the HyperdriveInterface api class."""

    def test_pool_config(self, local_hyperdrive_chain: LocalHyperdriveChain):
        """Checks that the Hyperdrive pool_config matches what is returned from the smart contract.

        All arguments are fixtures.
        """
        uri: URI | None = cast(HTTPProvider, local_hyperdrive_chain.web3.provider).endpoint_uri
        rpc_uri = uri if uri else URI("http://localhost:8545")
        abi_dir = "./packages/hyperdrive/src/abis"
        hyperdrive_contract_addresses: HyperdriveAddresses = local_hyperdrive_chain.hyperdrive_contract_addresses
        eth_config = EthConfig(artifacts_uri="not used", rpc_uri=rpc_uri, abi_dir=abi_dir)
        hyperdrive = HyperdriveInterface(eth_config, addresses=hyperdrive_contract_addresses)
        pool_config = smart_contract_read(hyperdrive.hyperdrive_contract, "getPoolConfig")
        assert pool_config == hyperdrive._contract_pool_config  # pylint: disable=protected-access

    def test_pool_info(self, local_hyperdrive_chain: LocalHyperdriveChain):
        """Checks that the Hyperdrive pool_info matches what is returned from the smart contract.

        All arguments are fixtures.
        """
        uri: URI | None = cast(HTTPProvider, local_hyperdrive_chain.web3.provider).endpoint_uri
        rpc_uri = uri if uri else URI("http://localhost:8545")
        abi_dir = "./packages/hyperdrive/src/abis"
        hyperdrive_contract_addresses: HyperdriveAddresses = local_hyperdrive_chain.hyperdrive_contract_addresses
        eth_config = EthConfig(artifacts_uri="not used", rpc_uri=rpc_uri, abi_dir=abi_dir)
        hyperdrive = HyperdriveInterface(eth_config, addresses=hyperdrive_contract_addresses)
        pool_info = smart_contract_read(hyperdrive.hyperdrive_contract, "getPoolInfo")
        for k,v in pool_info.items():
            print(f"{k:30} | {v}")
        print("\n ==== API ===")
        key_width = max(len(k) for k in dir(hyperdrive) if not k.startswith("_") and not k.startswith("pool_") and k != "current_block") + 2
        for k in dir(hyperdrive):  # sourcery skip: no-loop-in-tests
            if not k.startswith("_"):  # sourcery skip: no-conditionals-in-tests
                attr = getattr(hyperdrive, k)
                if not callable(attr) and not k.startswith("pool_") and k != "current_block":
                    if isinstance(attr, dict):
                        print(f"{k:{key_width}}")
                        attr_key_width = max(len(k) for k in attr.keys()) + 2
                        for attr_key, attr_val in attr.items():
                            print(f" | {attr_key:{attr_key_width}} | {attr_val}")
                    else:
                        print(f"{k:{key_width}} | {attr}")
        assert pool_info == hyperdrive._contract_pool_info  # pylint: disable=protected-access

    def test_checkpoint(self, local_hyperdrive_chain: LocalHyperdriveChain):
        """Checks that the Hyperdrive checkpoint matches what is returned from the smart contract.

        All arguments are fixtures.
        """
        uri: URI | None = cast(HTTPProvider, local_hyperdrive_chain.web3.provider).endpoint_uri
        rpc_uri = uri if uri else URI("http://localhost:8545")
        abi_dir = "./packages/hyperdrive/src/abis"
        hyperdrive_contract_addresses: HyperdriveAddresses = local_hyperdrive_chain.hyperdrive_contract_addresses
        eth_config = EthConfig(artifacts_uri="not used", rpc_uri=rpc_uri, abi_dir=abi_dir)
        hyperdrive = HyperdriveInterface(eth_config, addresses=hyperdrive_contract_addresses)
        checkpoint = smart_contract_read(
            hyperdrive.hyperdrive_contract, "getCheckpoint", hyperdrive.current_block_number
        )
        assert checkpoint == hyperdrive._contract_latest_checkpoint  # pylint: disable=protected-access

    def test_misc(self, local_hyperdrive_chain: LocalHyperdriveChain):
        """Placeholder for additional tests.

        These are only verifying that the attributes exist and functions can be called.
        All arguments are fixtures.
        """
        uri: URI | None = cast(HTTPProvider, local_hyperdrive_chain.web3.provider).endpoint_uri
        rpc_uri = uri if uri else URI("http://localhost:8545")
        abi_dir = "./packages/hyperdrive/src/abis"
        hyperdrive_contract_addresses: HyperdriveAddresses = local_hyperdrive_chain.hyperdrive_contract_addresses
        eth_config = EthConfig(artifacts_uri="not used", rpc_uri=rpc_uri, abi_dir=abi_dir)
        hyperdrive = HyperdriveInterface(eth_config, addresses=hyperdrive_contract_addresses)
        _ = hyperdrive.pool_config
        _ = hyperdrive.pool_info
        _ = hyperdrive.latest_checkpoint
        _ = hyperdrive.current_block
        _ = hyperdrive.current_block_number
        _ = hyperdrive.current_block_time
        _ = hyperdrive.spot_price
        _ = hyperdrive.get_max_long(FixedPoint(1000))
        _ = hyperdrive.get_max_short(FixedPoint(1000))
        # TODO: need an agent address to mock up trades
