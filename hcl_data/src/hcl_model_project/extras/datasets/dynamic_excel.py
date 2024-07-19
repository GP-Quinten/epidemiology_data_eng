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
            return pd.read_csv(load_path, **self._load_args)

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
                if file_name.startswith("Aiolos_HCL"):
                    # If the file name start with "Aiolos_HCL"
                    with sftp.open(f"{load_path}/{file_name}", "r") as file:
                        # Read the file
                        xls = pd.ExcelFile(file)
                        sheet_names = xls.sheet_names
                        sheet_data = {}
                        for sheet_name in sheet_names:
                            sheet_data[sheet_name] = pd.read_excel(
                                xls, sheet_name=sheet_name, **self._load_args
                            )
                        data[file_name] = sheet_data

        return data
