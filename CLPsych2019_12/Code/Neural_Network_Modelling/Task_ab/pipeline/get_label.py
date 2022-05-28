import csv
import os
cwd = os.getcwd()

with open('/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Code/Neural_Network_Modelling/Task_ab/pipeline/result.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i=0
    for row in spamreader:
        i+=1
        if i == 2:
            variable = row[0][-1]
            print(variable)

with open('/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Code/Neural_Network_Modelling/Task_ab/pipeline/label.txt', 'w') as f:
    if variable == '0':
        f.write('a: no risk')
    elif variable == '1':
        f.write('b: low risk')
    elif variable == '2':
        f.write('c: moderate risk')
    else:
        f.write('d: high risk')
