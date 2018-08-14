from abc import ABCMeta, abstractmethod


class QuickSort(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def select_pivot(self):
        return

    def partition_array(self):
        pass


class SortStrategy(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def sort(self):
        return


class NonDecreasingSortStrategy(SortStrategy):
    def sort(self):
        # TODO: implement
        pass


class NonIncreasingSortStrategy(SortStrategy):
    def sort(self):
        # TODO: implement
        pass


class QuickSortPivotFirst(QuickSort):
    def __init__(self, sortStrategy=NonDecreasingSortStrategy()):
        self.sortStrategy = sortStrategy

    def select_pivot(self):
        # TODO: implement
        pass


class QuickSortPivotEnd(QuickSort):
    def __init__(self, sortStrategy=NonDecreasingSortStrategy()):
        self.sortStrategy = sortStrategy

    def select_pivot(self):
        # TODO: implement
        pass

