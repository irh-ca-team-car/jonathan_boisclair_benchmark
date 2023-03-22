import torch

class A:
    def __init__(self) -> None:
        self.__target=None
   
    def setTarget(self,value):
        self.__target = value
        return value
    def getTarget(self):
        return self.__target 
    
    target=property(getTarget,setTarget,None)

a = A()


print(a.target)

a.target= 1