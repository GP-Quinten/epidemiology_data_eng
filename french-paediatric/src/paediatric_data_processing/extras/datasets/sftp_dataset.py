from typing import Any, Dict

import pandas as pd
from kedro.extras.datasets.pandas import CSVDataSet, ExcelDataSet
from kedro.io.core import PROTOCOL_DELIMITER, Version

# import paramiko

# import fsspec


class SFTPDataSet(CSVDataSet):
    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        version: Version = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ) -> None:
        # print("Filepath is: ")
        # print(filepath)

        super().__init__(
            filepath, load_args, save_args, version, credentials, fs_args
        )

    def _load(self) -> pd.DataFrame:
        load_path = str(self._get_load_path())

        print("Loading path: ...")
        print(load_path)
        if self._protocol == "file":
            # file:// protocol seems to misbehave on Windows
            # (<urlopen error file not on local host>),
            # so we don't join that back to the filepath;
            # storage_options also don't work with local paths
            return pd.read_csv(load_path, **self._load_args)

        load_path = f"{self._protocol}{PROTOCOL_DELIMITER}{load_path}"

        sftp = self._fs

        with sftp.open(load_path) as f:
            data = pd.read_csv(f, header=[0, 1], **self._load_args)

        return data

