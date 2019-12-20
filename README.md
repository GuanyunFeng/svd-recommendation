# 使用svd分解进行矩阵还原做推荐
## 使用数据集：
1.movielens数据集<br>
2.高斯随机生成两个矩阵，相乘得到低秩矩阵
## 测试方法：
取矩阵中的一个值，若预测值与真实值误差小于error=0.05认为还原成功。取多个值计算命中率hiting rate。<br>
## 算法
### SVD

SVD使用以下方式做矩阵分解:

![SVD](http://latex.codecogs.com/gif.latex?r_%7Bui%7D%20%3D%20%5Cmu%20&plus;%20b_u%20&plus;%20b_i%20&plus;%20p_u%20q_i)

LHS是预测打分。损失函数计算真实打分和预测打分的L2损失之和。参数更新时，梯度向最小化目标函数的方向下降。

### SVD++

SVD++算法和SVD是相似的，但包含了用户的*隐式反馈*。<br>
![SVD++](http://latex.codecogs.com/gif.latex?r_%7Bui%7D%20%3D%20%5Cmu%20&plus;%20b_u%20&plus;%20b_i%20&plus;%20%28p_u%20&plus;%20%5Cfrac%7B1%7D%7B%5Csqrt%7B%7CN%28u%29%7C%7D%7D%20%5Csum_%7Bj%20%5Cin%20N%28u%29%7D%20y_j%29%20q_i)

其中![用户隐式反馈](http://latex.codecogs.com/gif.latex?N%28u%29)是用户隐式反馈。<br>
在SVD++中,可以使用dual参数来决定是否包含项目的隐式反馈。公式可以被重新写做：
![dual SVD++](http://latex.codecogs.com/gif.latex?r_%7Bui%7D%20%3D%20%5Cmu%20&plus;%20b_u%20&plus;%20b_i%20&plus;%20%28p_u%20&plus;%20%5Cfrac%7B1%7D%7B%5Csqrt%7B%7CN%28u%29%7C%7D%7D%20%5Csum_%7Bj%20%5Cin%20N%28u%29%7D%20y_j%29%20%28q_i%20&plus;%20%5Cfrac%7B1%7D%7B%5Csqrt%7B%7CH%28i%29%7C%7D%7D%20%5Csum_%7Bj%20%5Cin%20H%28i%29%7D%20g_j%29)

其中![项目隐式反馈](http://latex.codecogs.com/gif.latex?H%28i%29)表示项目隐式反馈。
