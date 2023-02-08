from typing import List

def apply(samples, transforms:List):
    for x in transforms:
        samples = x(samples)
    return samples