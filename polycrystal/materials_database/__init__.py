"""Load YAML database of materials"""

import yaml

from .loaders import get_loader


class MaterialsDataBase:
    """Material Database

    Parameters
    ----------
    db_file: str or Path
        name of YAML database file
    """

    def __init__(self, db_file):
        self.db_file = db_file
        self.processes = dict()
        self._load_db()

    def _load_db(self):
        with open(self.db_file, "r") as f:
            db = yaml.safe_load_all(f)
            for doc in db:
                if "process" not in doc:
                    continue
                process, matls = doc['process'], doc['materials']
                self.processes[process] = {m['name']: m for m in matls}


    def list_processes(self):
        """List material processes"""
        return list(self.processes.keys())

    def list_materials(self, process):
        """List material names for given process

        Parameters
        ----------
        process: str
           name of material process
        """
        return list(self.processes[process].keys())

    def get_material(self, process, name):

        """Return material data

        Parameters
        ----------
        process: str
           name of material process
        name: str
           name of material to return

        Returns
        -------
        Instance
           instance of material class correpsonding to material process
        """
        matl_d = self.processes[process]
        Loader = get_loader(process)
        try:
            entry = matl_d[name]
        except:
            raise KeyError(f"material '{name}' not found in database")

        return Loader(entry)
