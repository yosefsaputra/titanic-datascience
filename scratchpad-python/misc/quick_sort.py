'''
Quick Sort
'''

import random


def partition(unsortedList, pivot_seq):
    i = 0
    j = len(unsortedList) - 1

    pivot = unsortedList[pivot_seq]

    while i < j:
        while unsortedList[i] < pivot and i < j:
            i += 1
        while unsortedList[j] >= pivot and i < j:
            j -= 1

        if i < j and unsortedList[i] > unsortedList[j]:
            tmp = unsortedList[i]
            unsortedList[i] = unsortedList[j]
            unsortedList[j] = tmp

    return unsortedList[:i], unsortedList[i:]


def quickSort(unsortedList):
    sortList = []

    pivot_seq = int(len(unsortedList) / 2)
    leftList, rightList = partition(unsortedList, pivot_seq)

    if len(leftList) > 1:
        leftList = quickSort(leftList)
    if len(rightList) > 1:
        rightList = quickSort(rightList)

    sortList.extend(leftList)
    sortList.extend(rightList)

    return sortList


# generatedNumbers = list(range(10))
generatedNumbers = [2, 0, 1]
# random.shuffle(generatedNumbers)
print(generatedNumbers)
generatedNumbers = quickSort(generatedNumbers)
print(generatedNumbers)
