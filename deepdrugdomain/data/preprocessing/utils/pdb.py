from Bio.PDB import PDBList
import os


def download_pdb(pdb, path):
    # Check if PDB file exists locally, else download it
    if not os.path.exists(path + pdb + '.pdb'):
        pdb_f = PDBList(verbose=False)
        pdb_f.retrieve_pdb_file(
            pdb, pdir=path, overwrite=False, file_format="pdb"
        )

        # Rename file to standard .pdb format from .ent
        os.rename(
            path + "pdb" + pdb + ".ent", path + pdb + ".pdb"
        )
        # Confirm the file has been downloaded
        if not any(pdb in s for s in os.listdir(path)):
            raise ValueError

    pdb_file = path + pdb + ".pdb"

    return pdb_file
