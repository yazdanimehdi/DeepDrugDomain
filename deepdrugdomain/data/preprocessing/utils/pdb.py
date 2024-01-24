from urllib.request import urlretrieve
from Bio.PDB import PDBList
import os
import re
import requests

URL_RE = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
def download_pdb(pdb, path):
    # Check if PDB file exists locally, else download it
    if re.match(URL_RE, pdb):
        identifier = pdb.split("/")[-1]
        response = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{identifier}")
        if response.status_code != 200:
            raise ValueError
        pdb_url = response.json()[0]["pdbUrl"]
        filename = os.path.join(path, identifier + ".pdb")

        urlretrieve(pdb_url, filename)
        return filename
    
    elif len(pdb) > 4:
        pdb = pdb[:4]

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
