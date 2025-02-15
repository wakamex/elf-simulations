"""Utility function for logging agent crash reports."""
from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
from collections import OrderedDict
from datetime import datetime, timezone
from enum import Enum
from traceback import format_tb
from types import TracebackType
from typing import TYPE_CHECKING, Any

import numpy as np
from agent0.hyperdrive.state import HyperdriveWallet, TradeResult, TradeStatus
from elfpy.utils import logs
from eth_typing import BlockNumber
from ethpy.base.errors import ContractCallException
from ethpy.hyperdrive.interface import (
    get_hyperdrive_checkpoint,
    get_hyperdrive_pool_info,
    process_hyperdrive_checkpoint,
    process_hyperdrive_pool_config,
    process_hyperdrive_pool_info,
)
from fixedpointmath import FixedPoint
from hexbytes import HexBytes
from numpy.random._generator import Generator as NumpyGenerator
from web3 import Web3
from web3.datastructures import AttributeDict, MutableAttributeDict
from web3.types import RPCEndpoint

if TYPE_CHECKING:
    from agent0.hyperdrive.agents import HyperdriveAgent
    from agent0.hyperdrive.state import HyperdriveMarketAction
    from elfpy import types
    from ethpy.hyperdrive.api import HyperdriveInterface


class ExtendedJSONEncoder(json.JSONEncoder):
    r"""Custom encoder for JSON string dumps"""

    # pylint: disable=too-many-return-statements
    # pylint: disable=too-many-branches
    def default(self, o):
        r"""Override default behavior"""
        if isinstance(o, set):
            return list(o)
        if isinstance(o, HexBytes):
            return o.hex()
        if isinstance(o, (AttributeDict, MutableAttributeDict)):
            return dict(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, FixedPoint):
            return str(o)
        if isinstance(o, NumpyGenerator):
            return "NumpyGenerator"
        if isinstance(o, datetime):
            return str(o)
        if isinstance(o, TracebackType):
            return format_tb(o)
        if isinstance(o, BaseException):
            return repr(o)
        if isinstance(o, Enum):
            return o.name
        try:
            return o.__dict__
        except AttributeError:
            pass
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


def setup_hyperdrive_crash_report_logging(log_format_string: str | None = None) -> None:
    """Create a new logging file handler with CRITICAL log level for hyperdrive crash reporting.

    In the future, a custom log level could be used specific to crash reporting.

    Arguments
    ---------
    log_format_string : str, optional
        Logging format described in string format.
    """
    logs.add_file_handler(
        logger=None,  # use the default root logger
        log_filename="hyperdrive_crash_report.log",
        log_format_string=log_format_string,
        delete_previous_logs=False,
        log_level=logging.CRITICAL,
    )


def build_crash_trade_result(
    exception: BaseException,
    agent: HyperdriveAgent,
    trade_object: types.Trade[HyperdriveMarketAction],
    hyperdrive: HyperdriveInterface,
) -> TradeResult:
    """Build the trade result object when a crash happens.

    Arguments
    ---------
    exception : Exception
        The exception that was thrown
    """
    trade_result = TradeResult(
        status=TradeStatus.FAIL,
        agent=agent,
        trade_object=trade_object,
    )

    # We first check if the exception came from a contract call
    # If it did, we fill various trade result data with custom data from
    # the exception
    trade_result.exception = exception
    if isinstance(exception, ContractCallException):
        trade_result.orig_exception = exception.orig_exception
        if exception.block_number is not None:
            trade_result.block_number = exception.block_number
        else:
            # Best effort to get the block it crashed on
            # We assume the exception happened in the previous block
            trade_result.block_number = hyperdrive.current_block_number - 1
        trade_result.contract_call = {
            "contract_call_type": exception.contract_call_type,
            "function_name_or_signature": exception.function_name_or_signature,
            "fn_args": exception.fn_args,
            "fn_kwargs": exception.fn_kwargs,
        }
        trade_result.raw_transaction = exception.raw_txn

    else:
        # Best effort to get the block it crashed on
        # We assume the exception happened in the previous block
        trade_result.block_number = hyperdrive.current_block_number - 1
        # We still build this structure so the schema stays the same
        trade_result.contract_call = {
            "contract_call_type": None,
            "function_name_or_signature": None,
            "fn_args": None,
            "fn_kwargs": None,
        }

    # We get the underlying contract info and convert them to human readable versions
    # Despite these being protected variables, we need low level access for crash reporting
    trade_result.block_timestamp = hyperdrive.web3.eth.get_block(trade_result.block_number).get("timestamp", None)
    # Pool config is static, so we can get it from the interface here
    trade_result.raw_pool_config = hyperdrive._contract_pool_config  # pylint: disable=protected-access

    # We wrap contract calls in a try catch to avoid crashing during crash report
    try:
        trade_result.raw_pool_info = get_hyperdrive_pool_info(
            hyperdrive.hyperdrive_contract, BlockNumber(trade_result.block_number)
        )
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Failed to get hyperdrive pool info in crash reporting: %s", repr(exc))
        trade_result.raw_pool_info = None

    try:
        if trade_result.block_timestamp is not None:
            trade_result.raw_checkpoint = get_hyperdrive_checkpoint(
                hyperdrive.hyperdrive_contract, hyperdrive.calc_checkpoint_id(trade_result.block_timestamp)
            )
        else:
            logging.warning("Failed to get block_timestamp in crash_reporting")
            trade_result.raw_checkpoint = None
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Failed to get hyperdrive checkpoint in crash reporting: %s", repr(exc))
        trade_result.raw_checkpoint = None

    # We call the conversion functions to convert them to human readable versions as well
    # TODO make process functions not affect state
    trade_result.pool_config = process_hyperdrive_pool_config(
        copy.deepcopy(trade_result.raw_pool_config), hyperdrive.hyperdrive_contract.address
    )

    if trade_result.raw_pool_info is not None:
        trade_result.pool_info = process_hyperdrive_pool_info(
            copy.deepcopy(trade_result.raw_pool_info),
            hyperdrive.web3,
            hyperdrive.hyperdrive_contract,
            BlockNumber(trade_result.block_number),
        )
    else:
        trade_result.pool_info = None

    if trade_result.raw_checkpoint is not None:
        trade_result.checkpoint_info = process_hyperdrive_checkpoint(
            copy.deepcopy(trade_result.raw_checkpoint),
            hyperdrive.web3,
            BlockNumber(trade_result.block_number),
        )
    else:
        trade_result.checkpoint_info = None

    trade_result.contract_addresses = {
        "hyperdrive_address": hyperdrive.hyperdrive_contract.address,
        "base_token_address": hyperdrive.base_token_contract.address,
    }

    # add additional information to the exception
    trade_result.additional_info = {
        "spot_price": hyperdrive.spot_price,
        "fixed_rate": hyperdrive.fixed_rate,
        "variable_rate": hyperdrive.variable_rate,
        "vault_shares": hyperdrive.vault_shares,
    }

    return trade_result


def log_hyperdrive_crash_report(
    trade_result: TradeResult,
    log_level: int | None = None,
    crash_report_to_file: bool = True,
    crash_report_file_prefix: str | None = None,
) -> None:
    # pylint: disable=too-many-arguments
    """Log a crash report for a hyperdrive transaction.

    Arguments
    ---------
    trade_result: TradeResult
        The trade result object that stores all crash information
    log_level: int | None
        The logging level for this crash report. Defaults to critical.

    Returns
    -------
    None
        This function does not return any value.
    """
    if log_level is None:
        log_level = logging.CRITICAL

    # If we're crash reporting, an exception is expected
    assert trade_result.exception is not None

    curr_time = datetime.utcnow().replace(tzinfo=timezone.utc)
    fn_time_str = curr_time.strftime("%Y_%m_%d_%H_%M_%S_Z")
    time_str = curr_time.isoformat()

    orig_traceback = None
    if trade_result.orig_exception is not None:
        orig_traceback = trade_result.orig_exception.__traceback__

    dump_obj = OrderedDict(
        [
            ("log_time", time_str),
            ("block_number", trade_result.block_number),
            ("block_timestamp", trade_result.block_timestamp),
            ("exception", trade_result.exception),
            ("orig_exception", trade_result.orig_exception),
            ("trade", _hyperdrive_trade_obj_to_dict(trade_result.trade_object)),
            ("contract_call", trade_result.contract_call),
            ("wallet", _hyperdrive_wallet_to_dict(trade_result.agent.wallet)),
            ("agent_info", _hyperdrive_agent_to_dict(trade_result.agent)),
            # TODO Once pool_info and pool_config are objects,
            # we need to add a conversion function to convert to dict
            ("pool_config", trade_result.pool_config),
            ("pool_info", trade_result.pool_info),
            ("checkpoint_info", trade_result.checkpoint_info),
            ("contract_addresses", trade_result.contract_addresses),
            ("additional_info", trade_result.additional_info),
            ("traceback", trade_result.exception.__traceback__),
            ("orig_traceback", orig_traceback),
            # NOTE if this crash report happens in a PR that gets squashed,
            # we loose this hash.
            ("commit_hash", _get_git_revision_hash()),
        ]
    )

    # We use ordered dict to ensure the outermost order is preserved
    logging_crash_report = json.dumps(dump_obj, indent=2, cls=ExtendedJSONEncoder)

    logging.log(log_level, logging_crash_report)

    # We print out a machine readable crash report
    if crash_report_to_file:
        # We add the machine readable version of the crash to the file
        # OrderedDict doesn't play nice with types
        dump_obj["raw_transaction"] = trade_result.raw_transaction  # type: ignore
        dump_obj["raw_pool_config"] = trade_result.raw_pool_config  # type: ignore
        dump_obj["raw_pool_info"] = trade_result.raw_pool_info  # type: ignore
        dump_obj["raw_checkpoint"] = trade_result.raw_checkpoint  # type: ignore
        dump_obj["anvil_dump_state"] = trade_result.anvil_state  # type: ignore
        # Generate filename
        if crash_report_file_prefix is None:
            crash_report_file_prefix = ""
        crash_report_dir = ".crash_report/"
        crash_report_file = f"{crash_report_dir}/{crash_report_file_prefix}{fn_time_str}.json"
        if not os.path.exists(crash_report_dir):
            os.makedirs(crash_report_dir)
        with open(crash_report_file, "w", encoding="utf-8") as file:
            json.dump(dump_obj, file, indent=2, cls=ExtendedJSONEncoder)


def _hyperdrive_wallet_to_dict(wallet: HyperdriveWallet) -> dict[str, Any]:
    """Helper function to convert hyperdrive wallet object to a dict keyed by token, valued by amount

    Arguments
    ---------
    wallet : HyperdriveWallet
        The HyperdriveWallet object to convert

    Returns
    -------
    dict[str, Any]
        A dict keyed by token, valued by amount
        In the case of longs and shorts, valued by a dictionary keyed by maturity_time and balance
    """

    # Keeping amounts here as FixedPoints for json to handle
    return {
        wallet.balance.unit.value: wallet.balance.amount,
        "longs": [
            {"maturity_time": maturity_time, "balance": amount.balance}
            for maturity_time, amount in wallet.longs.items()
        ],
        "shorts": [
            {"maturity_time": maturity_time, "balance": amount.balance}
            for maturity_time, amount in wallet.shorts.items()
        ],
        "lp_tokens": wallet.lp_tokens,
        "withdraw_shares": wallet.withdraw_shares,
    }


def _hyperdrive_trade_obj_to_dict(trade_obj: types.Trade[HyperdriveMarketAction]) -> dict[str, Any]:
    """Helper function to convert hyperdrive trade object to a dict

    Arguments
    ---------
    trade_obj: types.Trade[HyperdriveMarketAction]
        The trade object to convert

    Returns
    -------
    dict[str, Any]
        A dict ready to be converted to json
    """
    return {
        "market_type": trade_obj.market_type.name,
        "action_type": trade_obj.market_action.action_type.name,
        "trade_amount": trade_obj.market_action.trade_amount,
        "slippage_tolerance": trade_obj.market_action.slippage_tolerance,
        "maturity_time": trade_obj.market_action.maturity_time,
    }


def _hyperdrive_agent_to_dict(agent: HyperdriveAgent):
    return {"address": agent.checksum_address, "policy": agent.policy.name}


def _get_git_revision_hash() -> str:
    """Helper function for getting commit hash from git."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_anvil_state_dump(web3: Web3) -> str | None:
    """Helper function for getting anvil dump state"""
    result: str | None = None
    try:
        response = web3.provider.make_request(method=RPCEndpoint("anvil_dumpState"), params=[])
        result = response.get("result", False)
    except Exception:  # pylint: disable=broad-exception-caught
        # do nothing, this is best effort crash reporting
        pass
    return result
