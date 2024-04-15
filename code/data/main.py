

source_file_path1 = 'fr_en/training_attrs_1'
source_file_path2 = 'fr_en/training_attrs_2'
target_entity_path = 'out/entity_fr_en'

with open(source_file_path1, "r") as source_file1, open(source_file_path2, "r") as source_file2, open(target_entity_path, "w") as target_file:
    
    for line in source_file1:
        lines = line.split('\t')
        target_file.write(lines[0]+'\n')
    for line in source_file2:
        lines = line.split('\t')
        target_file.write(lines[0]+'\n')

