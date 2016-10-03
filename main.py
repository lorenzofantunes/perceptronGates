import csv
from perceptron import Perceptron

file = open('data.csv', 'r')
data = list(csv.reader(file, delimiter=',', quotechar='\n'))

inputs = []
targets = []

for row in data:
    targets.append(int(row[-1]))
    _input = []
    for value in row[:-1]:
        _input.append(int(value))
    inputs.append(_input)

perceptron = Perceptron()

perceptron.train(inputs, targets)

print("-1 -1 = " + str(perceptron.fit([-1, -1])))
print("-1 1 = " + str(perceptron.fit([-1, 1])))
print("1 -1 = " + str(perceptron.fit([1, -1])))
print("1 1 = " + str(perceptron.fit([1, 1])))
