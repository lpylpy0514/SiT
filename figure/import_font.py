import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = 'Times-New-Roman.ttf'
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()