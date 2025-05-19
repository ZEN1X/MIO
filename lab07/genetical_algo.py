from abc import ABC, abstractmethod
import numpy as np

class EncodingStyle(ABC):
    @abstractmethod
    def decode(self, chromosome: np.ndarray) -> float:
        pass

    @staticmethod
    def turn_to_bits(chromosome: np.ndarray) -> str:
        bits = "".join(str(int(b)) for b in chromosome)

        return bits


class BinaryEncoding(EncodingStyle):
    def __init__(self, length: int, x_min: float = 0.01, x_max: float = 1.0) -> None:
        super().__init__()
        self.length = length
        self.x_min = x_min
        self.x_max = x_max

    def decode(self, chromosome: np.ndarray) -> float:
        bits = self.turn_to_bits(chromosome)
        num = int(bits, 2)
        
        return self.x_min + num / (2**self.length-1) * (self.x_max - self.x_min)

class GrayEncoding(EncodingStyle):
    def __init__(self, length: int, x_min: float = 0.01, x_max: float = 1.0) -> None:
        super().__init__()
        self.length = length
        self.x_min = x_min
        self.x_max = x_max
    
    @staticmethod
    def gray_to_binary(gray: np.ndarray) -> np.ndarray:
        binary = gray.copy()
        for i in range(1, len(gray)):
            binary[i] ^= binary[i - 1]

        return binary

    def decode(self, chromosome: np.ndarray) -> float:
        bin_bits = self.gray_to_binary(chromosome)
        bits = self.turn_to_bits(bin_bits)
        num = int(bits, 2)

        return self.x_min + num / (2**self.length-1) * (self.x_max - self.x_min)

class SelectionStyle(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.generator = np.random.default_rng()

    @abstractmethod
    def select(self, individuals: list['Individual'], fitness: np.ndarray) -> list['Individual']:
        pass

class RouletteSelection(SelectionStyle):
    def __init__(self) -> None:
        super().__init__()

    def select(self, individuals: list['Individual'], fitness: np.ndarray) -> list['Individual']:
        vals = fitness - np.min(fitness) + 1e-9 # min-shifted
        probs = vals / np.sum(vals) # probabilities in range [0, 1]
        
        idx = self.generator.choice(len(individuals), size=len(individuals), p=probs)
        return [individuals[i].clone() for i in idx]
    
class ThresholdSelection(SelectionStyle):
    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma

    def select(self, individuals: list['Individual'], fitness: np.ndarray) -> list['Individual']:
        n = len(individuals) # total number of individuals
        cutoff = int(np.ceil(self.gamma / 100 * n)) # number of individuals that will reproduce
        top_idx = np.argsort(fitness)[-cutoff:] # top indices

        chosen = self.generator.choice(top_idx, size=n)
        return [individuals[i].clone() for i in chosen]


class Individual:
    def __init__(self) -> None:
        pass