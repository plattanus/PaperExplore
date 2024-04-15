s = set()
# 打开文件
with open('test.dat', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 去除行尾的换行符并分割成单词列表
        words = line.strip().split()
        # 输出每个单词
        for word in words:
            # print(word)
            s.add(word)

print(len(s))
