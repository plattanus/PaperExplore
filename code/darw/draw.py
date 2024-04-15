import matplotlib.pyplot as plt

x = [1000,2000,3000,4000,5000]
# y1 = [60.78,66.67,77.55,81.27,82.01]
# y2 = [66.23,76.85,80.11,84.71,83.01]
# y3 = [74.01,78.43,86.36,85.31,85.92]
# y4 = [70.24,78.42,84.92,85.36,84.74]
# y5 = [69.20,73.38,84.44,84.12,83.71]

y1 = [59.80,63.64,71.82,78.13,77.73]
y2 = [62.25,70.07,75.75,77.02,78.39]
y3 = [69.67,69.79,80.73,81.92,80.03]
y4 = [68.45,71.66,82.31,81.82,80.01]
y5 = [67.24,73.73,82.09,81.45,80.63]

# y1 = [59.11,64.50,70.42,73.07,77.00]
# y2 = [66.72,71.69,73.07,76.54,79.18]
# y3 = [70.01,72.20,82.22,81.61,81.44]
# y4 = [70.22,73.90,82.12,80.92,80.11]
# y5 = [68.28,75.79,81.21,80.04,80.13]

# 绘制折线图
plt.plot(x, y1, marker='o', label='k=10', color='blue')
plt.plot(x, y2, marker='o', label='k=20', color='green')
plt.plot(x, y3, marker='o', label='k=30', color='red')
plt.plot(x, y4, marker='o', label='k=40', color='orange')
plt.plot(x, y5, marker='o', label='k=50', color='purple')

# 添加标题和标签
plt.title('ja_en')
plt.xlabel('Candidate Set')
plt.ylabel('Hit@10')

# 添加图例
plt.legend()
plt.savefig('ja_en.png')
# 添加行和列的名称
# plt.xticks(x, ['A', 'B', 'C', 'D', 'E'])
# plt.yticks([60, 70, 80, 90, 100], ['60%', '70%', '80%', '90%', '100%'])

# 显示图形
plt.show()
