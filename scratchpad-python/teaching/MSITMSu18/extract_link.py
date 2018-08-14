import re
import os
import csv

directory = r'C:\Users\yosef\Desktop\Downloads\submissions'


def getUTID(string):
    return string.split("_")[0]


pat = '(\s*)(<meta http-equiv="Refresh" content="0; url=)(?P<link>.+)(/*">)'

link_dict = {}

for i, j, filenames in os.walk(directory):
    for filename in filenames:
        filePath = os.path.join(directory, filename)
        with open(filePath, 'r') as file:
            for line in file:
                match = re.match(pat, line)
                if match is not None:
                    link = match.group('link')
                    print(link)
                    link_dict[getUTID(filename)] = link

with open('link.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file, )
    for item in link_dict.items():
        csv_writer.writerow((item[0], item[1]))
