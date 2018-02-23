# Enter your code here. Read input from STDIN. Print output to STDOUT
import sys
import re


class SpecialDate:
    def __init__(self, year, month, date=1):
        self.year = int(year)
        self.month = int(month)
        self.date = int(date)

    def __eq__(self, date2):
        if self.year == date2.year and self.month == date2.month and self.date == date2.date:
            return True
        else:
            return False

    def __gt__(self, date2):
        if self.year > date2.year:
            return True
        elif self.year == date2.year:
            if self.month > date2.month:
                return True
            elif self.month == date2.month:
                if self.date > date2.date:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def __lt__(self, date2):
        if self.year < date2.year:
            return True
        elif self.year == date2.year:
            if self.month < date2.month:
                return True
            elif self.month == date2.month:
                if self.date < date2.date:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def __str__(self):
        string = ''
        string += '%4d-%02d-%02d' % (self.year, self.month, self.date)
        return string


def getInputData():
    return sys.stdin


def categorize(inputData):
    outputData = []

    fromDate = None
    toDate = None

    for seq, row in enumerate(inputData):
        if seq == 0:
            # Get requested interval
            pattern = r'(?P<fromyear>\d{4})-(?P<frommonth>\d+),\s*(?P<toyear>\d{4})-(?P<tomonth>\d+)'
            regex = re.match(pattern, row)
            if regex is not None:
                fromYear = regex.group('fromyear')
                fromMonth = regex.group('frommonth')
                toYear = regex.group('toyear')
                toMonth = regex.group('tomonth')

                fromDate = SpecialDate(int(fromYear), int(fromMonth))
                toDate = SpecialDate(int(toYear), int(toMonth))
        else:
            pattern = r'(?P<year>\d{4})-(?P<month>\d+)-(?P<date>\d+),\s*(?P<engagement>.[^,]+),\s*(?P<number>\d+)'
            regex = re.match(pattern, row)
            if regex is not None:
                year = regex.group('year')
                month = regex.group('month')
                date = regex.group('date')
                engagement = regex.group('engagement')
                number = int(regex.group('number'))
                dataDate = SpecialDate(int(year), int(month), int(date))

                if not (dataDate > toDate or dataDate < fromDate) and number > 0:
                    engagement = {
                        'name': engagement,
                        'number': number
                    }

                    isAdded = False
                    for date in outputData:
                        if date['date'].year == dataDate.year and date['date'].month == dataDate.month:
                            for i in date['engagements']:
                                if i['name'] == engagement['name']:
                                    i['number'] += engagement['number']
                                    isAdded = True
                            if not isAdded:
                                date['engagements'].append(engagement)
                                isAdded = True

                    if not isAdded:
                        dataPoint = {
                            'date': dataDate,
                            'engagements': [engagement]
                        }
                        outputData.append(dataPoint)

    outputData = sorted(outputData, key=lambda x: x['date'], reverse=True)
    for data in outputData:
        data['engagements'] = sorted(data['engagements'], key=lambda x: x['name'])

    return outputData


if __name__ == '__main__':
    inputData = getInputData()
    outputData = categorize(inputData)
    for i in outputData:
        string = ''
        string += '%d-%02d' % (i['date'].year, i['date'].month)
        for engagement in i['engagements']:
            string += ', %s, %d' % (engagement['name'], engagement['number'])
        print(string)
