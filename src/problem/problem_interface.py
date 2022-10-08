from abc import ABC, abstractmethod

# ProblemInterface is an abstract base class (ABC). The classes that
# inherit from ProblemInterface need to implement the abstract methods
# in order to be instantiable.
class ProblemInterface(ABC):

    @abstractmethod
    def task1(self, x0, alfa, h):
        pass

    @abstractmethod
    def task2(self):
        pass

    @abstractmethod
    def task3(self):
        pass

    @abstractmethod
    def task4_a(self):
        pass

    @abstractmethod
    def task4_b(self):
        pass

    @abstractmethod
    def task4_c(self):
        pass

    @abstractmethod
    def plot(self):
        pass
