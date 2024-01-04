"""Generic database utilities"""

from .interface import (
    TableWithBlockNumber,
    add_addr_to_username,
    add_username_to_user,
    close_session,
    drop_table,
    get_addr_to_username,
    get_latest_block_number_from_table,
    get_username_to_user,
    initialize_engine,
    initialize_session,
    initialize_duck,
    query_tables,
    _create_sequences_for_table,
)
from .schema import AddrToUsername, Base, UsernameToUser
