"""
PyMOL Visualization Script for Protein Binding Sites

This script allows visualization of protein structures in PyMOL, highlighting specific binding sites
based on coordinate ranges. The binding sites are defined by x, y, and z coordinate ranges, and
each site is colored uniquely for clear differentiation.

Examples:
    To use this script, first ensure that PyMOL is installed and running. Then, define the path to your
    PDB file and the binding site data as shown below:

    >>> pdb_path = "path_to_your_protein.pdb"
    >>> binding_sites_data = {
        ... "site1": {
        ...    "x_range": (5.0, 15.0),
        ...    "y_range": (20.0, 30.0),
        ...    "z_range": (30.0, 40.0),
        ...    "color": "red"
        ... },
        ... "site2": {
        ...    "x_range": (35.0, 45.0),
        ...    "y_range": (50.0, 60.0),
        ...    "z_range": (60.0, 70.0),
        ...    "color": "blue"
        ... }
    ... }
    >>> visualize_protein_with_custom_sites_and_surface(pdb_path, binding_sites_data)
"""

try:
    import pymol
    from pymol import cmd

except ImportError:
    print("PyMOL is not installed. Please install PyMOL to use this script.")
    raise
from typing import Tuple, Dict

# Initialize PyMOL
pymol.finish_launching()


def select_and_color_site(site_name: str, x_range: Tuple[float, float], y_range: Tuple[float, float],
                          z_range: Tuple[float, float], color_index: str) -> None:
    """
    Select and color atoms and surface in a specified site based on coordinate ranges.

    Parameters:
    site_name (str): Name of the binding site.
    x_range (Tuple[float, float]): Tuple of (min_x, max_x) coordinates.
    y_range (Tuple[float, float]): Tuple of (min_y, max_y) coordinates.
    z_range (Tuple[float, float]): Tuple of (min_z, max_z) coordinates.
    color_index (str): Color name or index for coloring atoms and surface in the site.

    Returns:
    None
    """
    cmd.select(
        site_name, f"br. within x, {x_range[0]}, {x_range[1]} and y, {y_range[0]}, {y_range[1]} and z, {z_range[0]}, {z_range[1]}")
    cmd.color(color_index, site_name)
    cmd.show("surface", site_name)
    cmd.set("surface_color", color_index, site_name)


def visualize_protein_with_custom_sites_and_surface(pdb_file: str, binding_sites_data: Dict[str, Dict[str, Tuple[float, float] or str]]) -> None:
    """
    Visualize a protein's surface with atoms in specified sites colored differently.

    Parameters:
    pdb_file (str): Path to the PDB file of the protein.
    binding_sites_data (Dict[str, Dict[str, Tuple[float, float] or str]]): 
        Dictionary with binding site data. Each key is a site name, and the value is a dictionary 
        containing coordinate ranges (x_range, y_range, z_range) and color for the site.

    Returns:
    None
    """
    cmd.load(pdb_file, "protein")
    cmd.show("surface", "protein")
    cmd.set("transparency", 0.5)  # Adjust the transparency as needed

    for site_name, data in binding_sites_data.items():
        select_and_color_site(
            site_name, data['x_range'], data['y_range'], data['z_range'], data['color'])

    cmd.zoom("protein")
