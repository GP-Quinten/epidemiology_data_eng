
from kedro.io import AbstractDataSet
from pymongo import MongoClient
from datetime import date, datetime

class MongoDBDataSet(AbstractDataSet):
    def __init__(self, credentials, database, collection):  
        self.database = database
        self.collection = collection
        self.client = None
        self.uri = credentials["uri"]

    def _load(self):
        if not self.client:
            self.client = MongoClient(self.uri)
        db = self.client[self.database]
        data = list(db[self.collection].find())
        return data

    def _save(self, data):
        # Find columns containing datetime.date objects
        date_columns = [col for col in data.columns if data[col].dtype == 'O' and isinstance(data[col].iloc[0], date)]

        for col in date_columns:
            data[col] = data[col].apply(lambda x: datetime.combine(x, datetime.min.time()))

        if not self.client:
            self.client = MongoClient(self.uri)
        db = self.client[self.database]
        data = data.to_dict(orient='records')

        # Clear the collection (remove all documents)
        db[self.collection].delete_many({})

        db[self.collection].insert_many(data)

    def _exists(self):
        if not self.client:
            self.client = MongoClient(self.uri)
        db = self.client[self.database]
        return self.collection in db.list_collection_names()

    def _describe(self):
        return dict(
            uri=self.uri,
            database=self.database,
            collection=self.collection
        )

    def _release(self):
        if self.client:
            self.client.close()

    def __del__(self):
        self._release()
