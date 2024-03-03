import matplotlib.pyplot as plt
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 假设我们有一些二维数据
x = [41,    77,    92,    33,  72,    76,    45,    47,   41,    18,    15,    11,    7,    5]
y = [0.811, 0.796, 0.788, 0.8, 0.777, 0.746, 0.766, 0.75, 0.725, 0.703, 0.554, 0.831, 0.813, 0.814]

colors = ['blue','blue','blue', 'c','c','c','c','c','c', 'green', 'green', 'green', 'green', 'green' ]

x1 = [41,    77,    92]
x2 = [33,  72,    76,    45,    47,   41]
x3 = [18,    15,    11,    7,    5]
y1 = [0.811, 0.796, 0.788]
y2 = [0.8, 0.777, 0.746, 0.766, 0.75, 0.725]
y3 = [0.703, 0.554, 0.831, 0.813, 0.814]
colors1 = ['blue','blue','blue']
colors2 = ['c','c','c','c','c','c']
colors3 = ['green', 'green', 'green', 'green', 'green']


names = ['SiT-Base', 'SiT-Small', 'SiT-Tiny', 'HiT-Base', 'HiT-Small', 'HiT-Tiny', 'HCAT', 'E.T.Track', 'LightTrack', 'ATOM', 'ECO', 'OSTrack-256', 'STARK-ST50', 'TransT']

print(len(x))
print(len(y))
print(len(names))
print(len(colors))
# 创建一个新的图形
plt.figure()

# 绘制散点图
# plt.scatter(x, y, c=colors)
plt.scatter(x1, y1, c=colors1, label="Our SiT Family")
plt.scatter(x2, y2, c=colors2, label="Other Real-time Trackers")
plt.scatter(x3, y3, c=colors3, label="Other Non-real-time Trackers")

# 添加图例
plt.legend()

ax = plt.gca()  # 获取当前的轴对象
ax.set_xlim([0, 100])  # 设置x轴范围
ax.set_ylim([0.5, 0.85])  # 设置y轴范围

plt.plot(x[0:3], y[0:3], color='blue', linestyle='--', label='Line connecting points')
# plt.plot(x[3:6], y[3:6], color='darksalmon', linestyle='--', label='Line connecting points')

# for i in range(len(x)):
#     plt.annotate(names[i], (x[i], y[i]))

# 在x=20处画一条竖直的虚线
plt.axvline(x=20, linestyle='--', color='grey')

# 设置x轴从0到20的范围的底色
plt.axvspan(0, 20, facecolor='whitesmoke', alpha=0.5)
# 设置x轴从0到20的范围的底色
plt.axvspan(20, 100, facecolor='beige', alpha=0.5)

# 设置标题和坐标轴标签
# plt.title('Model Speed on CPU（FPS）')
plt.xlabel('Model Speed on CPU(FPS)')
plt.ylabel('Success Rate(AUC)')

# 显示图形
# plt.show()

plt.savefig('picture1.svg', format='svg')