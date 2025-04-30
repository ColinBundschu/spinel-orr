
import asyncio
import datetime
from asyncio import Lock

import firebase_admin
from firebase_admin import firestore_async
from google.cloud.firestore import (AsyncClient, AsyncCollectionReference,
                                    AsyncDocumentReference)
from structlog.stdlib import BoundLogger

from constants import FIREBASE_CERT_PATH


def initalize_firestore(log: BoundLogger) -> AsyncClient:
    cred = firebase_admin.credentials.Certificate(FIREBASE_CERT_PATH)
    firebase_admin.initialize_app(cred)
    log.info('firebase init')
    return firestore_async.client()

class AsyncBatchDatabase:
    '''A class for batching database operations through firestore.
       This class operates as a singleton to ensure that only one firestore connection exists.
       "Instances" should be created for each use via an async context manager to ensure
       batch jobs are flushed. In theory, multiple instances of this class could be used
       simultaneously through different context managers, and the locks will allow them 
       to batch jobs together. Note that when each context exits it will flush the batch.
       This ensures that all jobs associated with a given context are completely flushed
       before the context exits.'''
    _instance = None
    _initialized = False  # flag to track if the singleton instance has been initialized

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncBatchDatabase, cls).__new__(cls)
        return cls._instance

    def __init__(self, log: BoundLogger, *, batch_size: int = 32):
        if self._initialized:
            return  # Skip initialization if instance already initialized

        self.log = log
        self.client = initalize_firestore(log)
        self.batch_size = batch_size
        self.batch = self.client.batch()
        self.count = 0
        self.lock = Lock()

        self._initialized = True  # set the flag to True after initialization

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        async with self.lock:
            await self._flush(flush_all = True)

    async def _flush(self, *, flush_all: bool = False):
        if self.batch_size < 1:
            raise ValueError(f'Batch size must be at least 1, but {self.batch_size} was given')

        if self.count >= self.batch_size or (flush_all and self.count > 0):
            self.log.debug(f'{self.count:<3} Flushing')
            await self.batch.commit()
            self.batch = self.client.batch()
            self.count = 0

    async def delete_doc(self, doc_ref: AsyncDocumentReference):
        tasks = [self.delete_collection(collection) async for collection in doc_ref.collections()]
        await asyncio.gather(*tasks)
        async with self.lock:
            self.log.debug(f'{self.count:<3} Deleting {doc_ref.path}')
            self.batch.delete(doc_ref)
            self.count += 1
            await self._flush()

    async def delete_collection(self, coll_ref: AsyncCollectionReference):
        tasks = [self.delete_doc(doc_ref) async for doc_ref in coll_ref.list_documents()]
        await asyncio.gather(*tasks)

    async def touch(self, doc_ref: AsyncDocumentReference):
        async with self.lock:
            self.log.debug(f'{self.count:<3} Touching {doc_ref.path}')
            self.batch.set(doc_ref, {'timestamp': datetime.datetime.now(datetime.timezone.utc)}, merge=True)
            self.count += 1
            await self._flush()

    async def set(self, doc_ref: AsyncDocumentReference, data: dict):
        async with self.lock:
            self.log.debug(f'{self.count:<3} Setting {doc_ref.path}')
            self.batch.set(doc_ref, data)
            self.count += 1
            await self._flush()

    async def make_parents(self, ref: AsyncDocumentReference | AsyncCollectionReference):
        tasks = []
        path_parts = ref.path.split('/')
        for i in range(1, len(path_parts)-1, 2):
            parent_path = '/'.join(path_parts[:i+1])
            parent_ref = self.client.document(parent_path)
            tasks.append(self.touch(parent_ref))
        await asyncio.gather(*tasks)

    async def update_field(self, collection_group_name: str, old_field_name: str, new_field_name: str):
        async with self.lock:
            collection_group_ref = self.client.collection_group(collection_group_name)
            async for doc in collection_group_ref.stream():
                data = doc.to_dict()
                if old_field_name in data:
                    data[new_field_name] = data.pop(old_field_name)
                    self.batch.set(doc.reference, data)
                    self.count += 1
                    await self._flush()
