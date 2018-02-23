'''
Merge Sort
'''

import random


def mergeList(leftList, rightList):
    i = 0
    j = 0
    sortedList = []

    while True:
        if leftList[i] <= rightList[j]:
            sortedList.append(leftList[i])
            i += 1
        else:
            sortedList.append(rightList[j])
            j += 1

        if i == len(leftList):
            sortedList.extend(rightList[j:])
            break
        elif j == len(rightList):
            sortedList.extend(leftList[i:])
            break

    return sortedList


def mergeSort(unsortedList):
    n = len(unsortedList)

    if n > 1:
        leftList = unsortedList[:int(n/2)]
        rightList = unsortedList[int(n/2):]

        leftList = mergeSort(leftList)
        rightList = mergeSort(rightList)

        sortedList = mergeList(leftList, rightList)

        return sortedList
    else:
        return unsortedList


generatedNumbers = list(range(10)) + list(range(10))
random.shuffle(generatedNumbers)
print(generatedNumbers)
generatedNumbers = mergeSort(generatedNumbers)
print(generatedNumbers)
