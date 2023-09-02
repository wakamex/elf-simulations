"""Script to generate typed web3.py classes for solidity contracts."""
from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import black
from jinja2 import Template
from pypechain.utilities.abi import (
    get_abi_items,
    get_param_name,
    get_structs_for_abi,
    is_abi_function,
    load_abi_from_file,
)
from pypechain.utilities.format import avoid_python_keywords, capitalize_first_letter_only
from pypechain.utilities.templates import setup_templates
from pypechain.utilities.types import solidity_to_python_type
from web3.types import ABIFunction


def main(abi_file_path: str, output_dir: str) -> None:
    """Generates class files for a given abi.

    Arguments
    ---------
    abi_file_path : str
        Path to the abi json file.

    output_dr: str
        Path to the directory to output the generated files.
    """

    # get names
    file_path = Path(abi_file_path)
    filename = file_path.name
    contract_name = os.path.splitext(filename)[0]

    # grab the templates
    contract_template, types_template = setup_templates()

    # render the code
    rendered_contract_code = render_contract_file(contract_name, contract_template, file_path)
    rendered_types_code = render_types_file(contract_name, types_template, file_path)

    # TODO: Add more features:
    # TODO:  events

    # Write the renders to a file
    types_output_file_path = Path(output_dir).joinpath(f"{contract_name}Types.py")
    contract_output_file_path = Path(output_dir).joinpath(f"{contract_name}Contract.py")
    with open(contract_output_file_path, "w", encoding="utf-8") as output_file:
        linted_code = black.format_file_contents(rendered_contract_code, fast=False, mode=black.mode.Mode())
        output_file.write(linted_code)
    with open(types_output_file_path, "w", encoding="utf-8") as output_file:
        linted_code = black.format_file_contents(rendered_types_code, fast=False, mode=black.mode.Mode())
        output_file.write(linted_code)


def render_contract_file(contract_name: str, contract_template: Template, abi_file_path: Path) -> str:
    """Returns a string of the contract file to be generated.

    Parameters
    ----------
    contract_template : Template
        A jinja template containging types for all structs within an abi.
    abi_file_path : Path
        The path to the abi file to parse.

    Returns
    -------
    str
        A serialized python file.
    """

    # TODO:  return types to function calls
    # Extract function names and their input parameters from the ABI
    abi_functions_and_events = get_abi_items(abi_file_path)

    function_datas = []
    for abi_function in abi_functions_and_events:
        if is_abi_function(abi_function):
            # TODO: investigate better typing here?  templete.render expects an object so we'll have to convert.
            name = abi_function.get("name", "")
            function_data = {
                # TODO: pass a typeguarded ABIFunction that has only required fields?
                # name is required in the typeguard.  Should be safe to default to empty string.
                "name": name,
                "capitalized_name": capitalize_first_letter_only(name),
                "input_names_and_types": get(abi_function, "inputs", include_types=True),
                "input_names": get(abi_function, "inputs", include_types=False),
                "outputs": get(abi_function, "outputs", include_types=True),
            }
            function_datas.append(function_data)
    # Render the template
    return contract_template.render(contract_name=contract_name, functions=function_datas)


def render_types_file(contract_name: str, types_template: Template, abi_file_path: Path) -> str:
    """Returns a string of the types file to be generated.

    Arguments
    ---------
    types_template : Template
        A jinja template containging types for all structs within an abi.
    abi_file_path : Path
        The path to the abi file to parse.

    Returns
    -------
    str
        A serialized python file.
    """

    abi = load_abi_from_file(abi_file_path)

    structs_by_name = get_structs_for_abi(abi)
    structs_list = list(structs_by_name.values())
    structs = [asdict(struct) for struct in structs_list]
    return types_template.render(contract_name=contract_name, structs=structs)

def get(function: ABIFunction, param_type, include_types: bool = True) -> list[str] | dict[str, str]:
    """Returns function inputs or outputs, optionally including types."""
    params = get_params(param_type, function)
    return params if include_types else [f"{k}: {v}" for k,v in params.items()]


def get_params(param_type, function) -> dict[str, str]:
    """Stringifies parameters."""
    parameters = function.get(param_type, [])
    formatted_params, anon_count = {}, 0
    for param in parameters:
        name = get_param_name(param)
        if name is None or name == "":
            name = f"{param_type[:-1]}{anon_count}"
            param["name"] = name
            anon_count += 1
        formatted_params[avoid_python_keywords(name)] = solidity_to_python_type(param.get("type"))
    return formatted_params


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <path_to_abi_file> <contract_address> <output_file>")
    else:
        # TODO: add a bash script to make this easier, i.e. ./pypechain './abis', './build'
        # TODO: make this installable so that other packages can use the command line tool
        main(sys.argv[1], sys.argv[2])
