import matplotlib.pyplot as plt

# 示例数据
x = [0.332, 0.322, 0.109, 0.714,
     1.312, 1.222, 1.009, 1.653]
y = [3.659, 3.489, 3.587, 3.551,
     1.653, 1.441, 1.667, 0.891]
colors = ['red', 'green', 'orange', 'blue',
          'red', 'green', 'orange', 'blue']
labels = ['(0.332,3.659)', '(0.322,3.489)', '(0.109,3.587)', '(0.714,3.551)',
          '(1.312,1.653)', '(1.222,1.441)', '(1.009,1.667)', '(1.653,0.891)']

# 创建图形
# plt.figure(figsize=(8, 6))

# 绘制点图并标注颜色和标签
labeled_points = set()  # 用于记录已经标记过的点
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], label=labels[i])
    if labels[i] not in labeled_points:  # 检查是否已经标记过该点
        plt.text(x[i] + 0.01, y[i], labels[i], fontsize=8)
        labeled_points.add(labels[i])  # 将该点添加到已标记集合中

# 在A点和B点之间绘制直线
# plt.plot([x[0], x[1]], [y[0], y[1]], color='black', linestyle='--', linewidth=1)

# 添加标题和标签
plt.title('Alignment of entities under the influence of BTM')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图例
plt.legend()

plt.savefig('com.png')
# 显示图形
plt.show()
