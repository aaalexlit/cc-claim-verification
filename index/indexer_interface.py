import abc
from haystack import Document


class IndexerInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'write_documents') and
                callable(subclass.write_documents) and
                hasattr(subclass, 'retrieve_matches_for_a_phrase') and
                callable(subclass.retrieve_matches_for_a_phrase) and
                hasattr(subclass, 'retrieve_matches_for_phrases') and
                callable(subclass.retrieve_matches_for_phrases) or
                NotImplemented)

    @abc.abstractmethod
    def write_documents(self, docs: [Document]):
        """Index documents"""
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve_matches_for_a_phrase(self, phrase: str, top_k: int):
        """Extract text from the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve_matches_for_phrases(self, phrases: [str], top_k: int):
        """Extract text from the data set"""
        raise NotImplementedError
