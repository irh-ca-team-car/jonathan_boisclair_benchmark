import math
class Batch:
    def of(dataset, size=8):
        return Batch(dataset, size=size)
    def __init__(self, dataset, size=8) -> None:
        self.dataset = dataset
        self.size = size
    def __len__(self):
        return int(math.ceil(float(len(self.dataset)) / self.size))
    def __getitem__(self,index):
        if isinstance(index,slice):
            values=[]
            if index.step is not None:
                values = [v for v in range(index.start,index.stop,index.step)]
            else:
                values = [v for v in range(index.start,index.stop)]
            values = [v for v in values if v < len(self.dataset)]
            return [self.__getitem__(v) for v in values]
        return self.dataset[index*self.size:index*self.size+self.size]
        