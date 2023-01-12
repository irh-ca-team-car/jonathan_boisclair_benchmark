from typing import List
from ..Sample import Sample

class DetectionDataset:
    def __init__(self) -> None:
        pass
    def classesList() -> List[str]:
        return []
    def getId(str: str):
        return 0
    def getName(id=None):
        return str(id)
    def isBanned(nameOrId):
        return False
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index: int) -> Sample:
        return Sample()