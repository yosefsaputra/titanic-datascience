from abc import ABCMeta, abstractmethod


class QuickSort(object):
    __meta__ = ABCMeta

    def __init__(self):
        self.pivot = 0

    @abstractmethod
    def select_pivot(self, array):
        return

    def partition_array(self, array):
        return array[0:self.pivot], array[self.pivot:]


class SortStrategy(object):
    __meta__ = ABCMeta

    @abstractmethod
    def sort(self):
        return


class NonIncreasingStrategy(SortStrategy):
    def sort(self):
        # implement
        return

class NonDecreasingStrategy(SortStrategy):
    def sort(self):
        # implement
        return


class QSPivotFirst(QuickSort):
    def __init__(self, strategy=NonIncreasingStrategy()):
        self.strategy = strategy

    def select_pivot(self, array):
        # implement
        self.pivot = 0


class QSPivotEnd(QuickSort):
    def select_pivot(self, array):
        # implement
        self.pivot = len(array) - 1


if __name__ == "__main__":
    array = list()
    qs = QSPivotFirst(strategy=NonDecreasingStrategy())
