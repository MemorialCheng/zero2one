**BI数据分析**

# 1 matplotlib
[matplotlib官网入口](https://matplotlib.org/)   
  

- plt.plot(x,y)  # 折线图  
- plt.scatter(x,y)  # 散点图  
- plt.bar(x, height, width=0.8)  # 绘制纵向条形图  
- plt.barh(y, width, height=0.8)  # 绘制横向条形图  
- plt.hist(data, num_bins)  # 直方图,num_bins组数

其他绘图工具：
1. [echarts官网入口](https://echarts.apache.org/examples/zh/index.html)

2. plotly:可视化工具中的github,相比于matplotlib更加简单,图形更加漂亮,同时兼容matplotlib和pandas。

    使用用法:简单,照着文档写即可。
   
    文档地址: https://plot.ly/python/


# 2 numpy

# 3 pandas

# 4 sklearn
[sklearn官网入口](https://scikit-learn.org/stable/)  
## 4.1 决策树
模块sklearn.tree
sklearn中决策树的类都在”tree“这个模块之下。这个模块总共包含五个类

|    API   |Description                          |
|----------------|-------------------------------|
|tree.DecisionTreeClassifier|分类树|
|tree.DecisionTreeRegressor|回归树|
|tree.export_graphviz|将生成的决策树导出为DOT格式，画图专用|
|tree.ExtraTreeClassifier|高随机版本的分类树|
|tree.ExtraTreeRegressor|高随机版本的回归树|



