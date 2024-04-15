
target_entity_path = 'out/entity_zh_en'
expected_value = 'expected_value'
target_path = 'out/com_zh_en'

arr = []
with open(target_entity_path, "r") as entity_path, open(expected_value, "r") as expected, open(target_path, "w") as path:
    
    for line in entity_path:
        arr.append(line)
    for line in expected:
        arr.append(line)
    for a in arr:
        path.write()