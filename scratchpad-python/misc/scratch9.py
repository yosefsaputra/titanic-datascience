import random
import time


def change(a, b, c, d, e):
    if a+b+c+d+e <= 1:
        return 1
    else:
        return 0


def change1(a, b, c, d, e):
    solutions = getSolutionSequences(5)

    correct = False
    for solution in solutions:
        if [a, b, c, d, e] == solution:
            correct = True

    if correct:
        return c

    if a == 0 and b == 0:
        return 1
    else:
        return 0


def getSolutionSequences(n):
    solution = []
    solution.append([1 if i % 3 == 0 else 0 for i in range(n)])
    solution.append([1 if i % 3 == 0 else 0 for i in range(1, n+1)])
    solution.append([1 if i % 3 == 0 else 0 for i in range(2, n+2)])
    return solution


n = 99
sequences = [
    [1 for i in range(n)],
    [0 for i in range(n)],
    [random.randint(0, 1) for i in range(n)],
    [random.randint(0, 1) for i in range(n)],
    [random.randint(0, 1) for i in range(n)],
    [random.randint(0, 1) for i in range(n)],
    [random.randint(0, 1) for i in range(n)],
    [random.randint(0, 1) for i in range(n)],
    [random.randint(0, 1) for i in range(n)],
    [random.randint(0, 1) for i in range(n)],
]

string = pString = ''
noOfSequence = 1

startTime = time.time()
# for sequence in sequences:
while noOfSequence < 2:
    print('%d : ' % noOfSequence, end='')
    sequence = [1 for i in range(n)]
    cSequence = sequence
    convergeCounter = 0
    unendingCounter = 0
    while convergeCounter < 10000 and unendingCounter < 1000000000:
        x = random.randint(0, n - 1)

        result = change(cSequence[x-2],
                        cSequence[x-1],
                        cSequence[x],
                        cSequence[x+1 if x+1 < n-1 else x+1-n],
                        cSequence[x+2 if x+2 < n-1 else x+2-n]
                        )
        cSequence[x] = result

        string = '%s' % cSequence
        if string != pString:
            print(cSequence)
            pString = string
            convergeCounter = 0
        else:
            convergeCounter += 1
        unendingCounter += 1

    # print(cSequence)
    result = False
    for solution in getSolutionSequences(n):
        if cSequence == solution:
            result = True
            break

    if not result:
        print(False)
        print(sequence)
    else:
        print('')

    noOfSequence += 1

print(time.time() - startTime)
