

source_file_path1 = 'fr_en/training_attrs_1'
source_file_path2 = 'fr_en/training_attrs_2'
target_file_path = 'out/mine_fr_en'

datacounter1 = {}
datacounter2 = {}
with open(source_file_path1, "r") as source_file:
    for line in source_file:
        lines = line.split('\t')
        for i in range(1, len(lines)):
            if lines[i] in datacounter1:
                datacounter1[lines[i]] += 1
            else:
                datacounter1[lines[i]] = 1
with open(source_file_path2, "r") as source_file:
    for line in source_file:
        lines = line.split('\t')
        for i in range(1, len(lines)):
            if lines[i] in datacounter2:
                datacounter2[lines[i]] += 1
            else:
                datacounter2[lines[i]] = 1


index = 1
datajson = {}
with open(source_file_path1, "r") as source_file1, open(source_file_path2, "r") as source_file2, open(target_file_path, "w") as target_file:
    
    for line in source_file1:
        lines = line.split('\t')
        datajson[lines[0]] = index
        index += 1
        s = ''
        for i in range(1, len(lines)):
            if s:
                s += ' '
            # if datacounter1[lines[i]] > 2000 or datacounter1[lines[i]] < 35:
            #     continue
            s += lines[i]        
        target_file.write(s)
    for line in source_file2:
        lines = line.split('\t')
        datajson[lines[0]] = index
        index += 1
        s = ''
        for i in range(1, len(lines)):
            if s:
                s += ' '
            # if datacounter2[lines[i]] > 2000 or datacounter2[lines[i]] < 35:
            #     continue
            s += lines[i]        
        target_file.write(s)


print(datajson)