from typing import Any, Dict

import pandas as pd
import paramiko
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.io import Version
from kedro.io.core import Version


class SFTPExcelLoader(CSVDataSet):
    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        version: Version = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ) -> None:
        # print("Filepath: ...")
        # print(filepath)

        super().__init__(
            filepath=filepath,
            load_args=load_args,
            save_args=save_args,
            version=version,
            credentials=credentials,
            fs_args=fs_args,
        )
        self.credentials = credentials

    def _load(self) -> Dict[str, pd.DataFrame]:
        load_path = str(self._get_load_path())
        if self._protocol == "file":
            # File protocol, load locally
            return pd.read_csv(
                load_path, sep=";", quotechar='"', encoding="latin1", **self._load_args
            )

        data = {}

        with paramiko.Transport(
            (self.credentials["host"], self.credentials["port"])
        ) as transport:
            transport.connect(
                username=self.credentials["username"],
                password=self.credentials["password"],
            )
            sftp = paramiko.SFTPClient.from_transport(transport)

            file_list = sftp.listdir(load_path)

            for file_name in file_list:
                # Vérifiez si le nom du fichier contient " - I"
                if " - I" in file_name and file_name.endswith("_cleaned.csv"):
                    excel_name = file_name  # Utilisez le nom complet de l'Excel avec l'extension ".csv"

                    # Faites ce que vous devez faire avec les informations extraites
                    with sftp.open(f"{load_path}/{file_name}", "r") as file:
                        sheet_data = pd.read_csv(
                            file,
                            sep=";",
                            quotechar='"',
                            encoding="latin1",
                            **self._load_args,
                        )

                        # Utilisez les informations extraites pour stocker les données ou effectuer d'autres opérations
                        data[excel_name] = sheet_data
        return data
