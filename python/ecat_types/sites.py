from abc import ABC, abstractmethod

from ecat_types.attach_site import AttachSite

class Sites(ABC):
    @abstractmethod
    def H_spec(self, i: int) -> AttachSite:
        raise NotImplementedError()
    
    @abstractmethod
    def matching_sites(self, key: int | str) -> list[AttachSite | list[AttachSite]]:
        raise NotImplementedError()
