from typing import Any, Dict

import pandas as pd
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.io.core import PROTOCOL_DELIMITER, Version


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

        """ 
        data = pd.read_csv(
            load_path, storage_options=self._storage_options, **self._load_args
        ) #urlopen error unknown url type: sftp
        """
        with sftp.open(load_path) as f:
            data = pd.read_csv(f, **self._load_args)

        return data

    """
    def _load(self) -> pd.DataFrame:
        load_path = str(self._get_load_path())
        if self._protocol == "sftp":
            transport = paramiko.Transport(
                (self._credentials.host, self._credentials.port)
            )
            transport.connect(
                username=self._credentials.username, password=self._credentials.password
            )
            sftp = paramiko.SFTPClient.from_transport(transport)

            with sftp.open(load_path) as f:
                data = pd.read_csv(
                    f, storage_options=self._storage_options, **self._load_args
                )
            sftp.close()
            return data
        else:
            raise ValueError("Protocol should be sftp")
    """
