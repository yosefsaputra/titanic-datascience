# Enter your code here. Read input from STDIN. Print output to STDOUT
import sys
import re
import datetime


def getInputData():
    '''
    To get input data from stdin
    :return: TextIO
    '''
    return sys.stdin


def categorize(inputData):
    '''
    From input data with specific format, this function parse the data. It gets:
    - start and end date interval
    - parse engagement data
    Then, categorize it by month. Engagements in a month is sorted by engagement category.

    :param inputData: Input Data Format:
                      - line 1: <start-date(yyyy-mm)>, <end-date(yyyy-mm)>
                      - line 2: <empty>
                      - line 3 & onwards: <date(yyyy-mm-dd)>, <engagement-category>, <number-of-engagement>
    :type inputData: TextIO

    :return: dict outputData
    '''
    # Initialize outputData as dictionary. Key: year-month. Value: list of engagement
    outputData = {}

    fromDate = None
    toDate = None

    # Going through each line
    for seq, row in enumerate(inputData):

        # Get Start and End of Interval from the first line
        if seq == 0:
            pattern = r'(?P<fromyear>\d{4})\s*(-\s*(?P<frommonth>\d+)\s*)*(-\s*(?P<fromdate>\d+)\s*)*,\s*(?P<toyear>\d{4})\s*(-\s*(?P<tomonth>\d+))*(-\s*(?P<todate>\d+)\s*)*'
            regex = re.match(pattern, row)
            if regex is not None:
                fromYear = regex.group('fromyear')
                try:
                    fromMonth = regex.group('frommonth')
                except Exception as e:
                    fromMonth = 1
                toYear = regex.group('toyear')
                try:
                    toMonth = regex.group('tomonth')
                except Exception as e:
                    toMonth = 1
                fromDates = None
                toDates = None
                try:
                    fromDates = regex.group('fromdate')
                except Exception as e:
                    fromDates = 1
                try:
                    toDates = regex.group('todate')
                except Exception as e:
                    toDates = 1

                try:
                    if fromDates is not None and fromMonth is not None:
                        fromDate = datetime.date(int(fromYear), int(fromMonth),
                                                 int(fromDates))  # start date, month are specified
                    elif fromMonth is not None:
                        fromDate = datetime.date(int(fromYear), int(fromMonth), 1)  # start date is unspecified
                    else:
                        fromDate = datetime.date(int(fromYear), 1, 1)  # start date, month are unspecified
                    if toDates is not None and toMonth is not None:
                        toDate = datetime.date(int(toYear), int(toMonth), int(toDates))  # end date, month are specified
                    elif toMonth is not None:
                        toDate = datetime.date(int(toYear), int(toMonth), 1)  # end date is unspecified
                    else:
                        toDate = datetime.date(int(toYear), 1, 1)  # start date, month are unspecified
                except ValueError as e:
                    print('ValueError: start or end date interval cannot be constructed because the date is invalid')
                    break
        else:
            # Parse engagement data
            pattern = r'(?P<year>\d{4})\s*-\s*(?P<month>\d+)\s*(-\s*(?P<date>\d+)\s*)*,\s*(?P<engagement>.[^,]+)\s*,\s*(?P<number>\d+)'
            regex = re.match(pattern, row)
            if regex is not None:  # if not according to pattern, pass
                year = regex.group('year')
                month = regex.group('month')
                try:
                    date = regex.group('date')
                except Exception as e:
                    date = 1
                engagement = regex.group('engagement')
                number = int(regex.group('number'))

                try:
                    dataDate = datetime.date(int(year), int(month), int(date))
                except ValueError as e:
                    print('ValueError: date cannot be constructed because the date is invalid')
                    continue

                if (toDate > dataDate > fromDate) and number > 0:

                    keyDate = datetime.date(dataDate.year, dataDate.month, 1)
                    if keyDate not in outputData:
                        outputData[keyDate] = {
                            engagement: number
                        }
                    else:
                        if engagement not in outputData[keyDate]:
                            outputData[keyDate][engagement] = number
                        else:
                            outputData[keyDate][engagement] += number
    return outputData


def printByMonth(outputData):
    '''
    To print output data to console with a specific format.
    :param outputData: engagement information by month
    :type outputData: dict
    :return: None
    '''
    for key in sorted(outputData.keys(), reverse=True):
        string = '%d-%02d' % (key.year, key.month)
        for i in sorted(outputData[key].keys()):
            print(i)
            if outputData[key][i] > 0:
                string += ', %s, %d' % (i, outputData[key][i])
        print(string)


if __name__ == '__main__':
    inputData = getInputData()
    outputData = categorize(inputData)
    printByMonth(outputData)
