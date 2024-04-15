
a1 = 'voca.txt'
a2 = 'model/expected_value.txt'
a3 = 'except'

arr1 = []
arr2 = []
with open(a1, "r") as b1, open(a2, "r") as b2, open(a3, "w") as b3:
    
    for line in b1:
        lines = line.split('\t')
        arr1.append(lines[1])
    for line in b2:
        arr2.append(line)
    for i in range(len(arr2)):
        b3.write(arr1[i][0:-1] + ' ' + arr2[i])