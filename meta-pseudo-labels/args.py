from dataclasses import dataclass, field


@dataclass
class Args:
    name: str
    data_path: str = field(default="./dataset")
    dataset: str = field(default="cifar10")
    
    
