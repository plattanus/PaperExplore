import numpy as np

# 读取 theta 文件
theta_file = "k10.pz"
with open(theta_file, "r") as file:
    theta_data = [[float(value) for value in line.strip().split()] for line in file]

theta_matrix = np.array(theta_data)

# 打印 theta 矩阵
print("Theta 矩阵:")
print(theta_matrix)

# 读取 phi 文件
phi_file = "k10.pw_z"
with open(phi_file, "r") as file:
    phi_data = [[float(value) for value in line.strip().split()] for line in file]

phi_matrix = np.array(phi_data)

# 打印 phi 矩阵
print("Phi 矩阵:")
print(phi_matrix)


# import numpy as np

# 假设有文档的主题分布 theta 和主题的词语分布 phi
# theta = np.array([0.2, 0.3, 0.5])  # 假设文档属于三个主题的概率分别为 0.2, 0.3, 0.5
# phi = np.array([[0.1, 0.2, 0.3, 0.2], [0.2, 0.3, 0.4, 0.2], [0.3, 0.4, 0.5, 0.2]])  # 假设三个主题的词语分布
theta = theta_matrix[0]
phi = phi_matrix
# 计算数学期望
expected_value = np.dot(theta, phi)
print("主题的数学期望:", expected_value)
np.savetxt('expected_value.txt', expected_value)