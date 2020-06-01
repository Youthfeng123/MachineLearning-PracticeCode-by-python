#numpy 使用笔记

#导入程序包
import numpy as np

array=np.array([[1,2,3],[4,5,6]])#创建一个矩阵/数组

print(array)
print('number of dimention:',array.ndim)  #维数
print('The shape:',array.shape) #形状
print('size:',array.size)  #元素总数


a=np.array([2,23,4],dtype=np.int64)  #定义数据形式  int32,float64,float32
print(a.dtype)
b=np.array([[2,23,4],
         [2,56,7]])  #定义二维矩阵
print(b)

zero=np.zeros((3,4))  #参数是形状
print(zero)

ones=np.ones((3,4),dtype=np.int)
print(ones)

empty=np.empty((3,4))
print(empty)

arange=np.arange(10,20,2)   #就不包括二十  第三个是步长
print(arange)

arange2=np.arange(12).reshape((3,4))  #默认从零开始 reshape可定义形状
print(arange2)

linespace=np.linspace(1,10,20)  #第三个参数是段数，会自动匹配步长
#print(linespace)

#基础运算

a=np.array([10,20,30,40])
b=np.arange(4)
print(a,b)
c=a-b  #加法同理
print(c)

#乘法
c=b**2  #元素去平方
print(c)

#函数运算
c=10*np.sin(a)  #参数是一个向量 cos tan同理
print(c)

#比较大小
print(b<3)

#矩阵的运算

a=np.array([[1,1],
           [0,1]])
b=np.arange(4).reshape((2,2))

#对应逐个相乘
c=a*b
print(c)
#矩阵乘法
c_dot=np.dot(a,b)#  c_dot=a.dot(b)  等价
print(c_dot)

random=np.random.random((2,4))  #随机生成0~1的数字
print(random)
print(np.sum(random,axis=1),np.min(random),np.max(random))  #求矩阵中的最值  求和

#axis=1:在行数中求值  axis=0:在列数中求值  几乎所有方法都有这个参数

#  索引

A=np.arange(2,14).reshape((3,4))
print(A)
print(np.argmin(A))  #计算最小值的索引
print(np.argmax(A))
#求平均值
print(np.mean(A))  #A.mean()  中位数：median
# 累加：cumsum(前几位相加的值)
print(np.cumsum(A))
#累差
print(np.diff(A))
#找出非零的数   麻烦 不管
print(np.nonzero(A))
print(np.sort(A))#排序  逐行排序  由小到大


#转置
print(np.transpose(A)) #A.T等价
print(np.clip(A,5,9))  #大于9的数变成9 小于5的数变成5


#索引  和列表差不多 从0开始
A=np.arange(3,15)
print(A)
print(A[3])
B=np.arange(3,15).reshape((3,4))
print(B)
print(B[0][3])  #B[0,3]等价
print(B[2:])  #第二行的所有数
print(B[:,2])  #第二列  输出的是行向量
print(B[1,1:3])  #第一行的 [1,3)列


for row in B:  #row是行  但是没有列的迭代
    print(row)
#列的迭代示例

for column in B.T:  #转置以后就变成列了
    print(column)


print(B.flatten())#向量化
for item in B.flat:  #对项目的迭代
    print(item)


#矩阵的合并

A=np.array([1,1,1])
B=np.array([2,2,2])
C=np.vstack((A,B))
D=np.hstack((A,B))
print(C)  #上下合并
print(D)#左右合并

print(A.T.shape)  #转置无法变成列向量

print(A[:,np.newaxis])  #增加了一个维度  要这么变！

C=np.concatenate((A,A,B))  #多个进行合并 axis=1横向合并
print(C)


#分割array

A=np.arange(12).reshape((3,4))
print(A)
#如果不能整除会报错啦！！！
print(np.split(A,2,axis=1))  #按照列分成两块 np.hsplit(A,2)

print(np.split(A,3,axis=0))  #按行分割成三块 np.vsplit(A,3)

#不等量的分割
print(np.array_split(A,3,axis=1))


#赋值,copy 和deep copy
a=np.arange(4)
b=a
c=a
d=b
a[0]=11
print(a,b)  #b就是完完全全的a  a变b也变 c也同理
print(b is a)
print(d is a)  #全都是a

#不关联的拷贝
b=a.copy()   #deep copy

