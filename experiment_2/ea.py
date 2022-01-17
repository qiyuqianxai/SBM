# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import time
from settings import get_energy_tp_matrix,get_center_node
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
"""
    该案例展示了一个带等式约束的连续型决策变量最大化目标的单目标优化问题。
    该函数存在多个欺骗性很强的局部最优点。
    min f = (x1 + x2 + x3)**2 + x1 + x2 + x3
    s.t.
    x1, x2, x3, x100 = 0 or 1
    sum (x1 + ... + x100) = 50
"""


# node_num = 15
# select_num = 10
# b_ratio = 0.9
#
# cp_matrix = np.ones(node_num)
# tp_matrix = get_energy_tp_matrix(node_num)
# ef_matrix = np.zeros(node_num)

class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self,cp_matrix,tp_matrix,ef_matrix,node_num,select_num,b_ratio):
        name = 'select nodes' # 初始化name（函数名称，可以随意设置）
        self.cp_matrix = cp_matrix
        self.tp_matrix = tp_matrix
        self.ef_matrix = ef_matrix
        self.node_num = node_num
        self.select_num = select_num
        self.b_ratio = b_ratio
        M = 1 # 初始化M（目标维数）
        maxormins = [-1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = self.node_num # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim # 决策变量下界
        ub = [1] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop): # 目标函数
        Vars = pop.Phen # 得到决策变量矩阵
        xall = 0
        x1 = 0 # cp_energy
        x2 = 0 # ef
        for i in range(self.node_num):
            xall = xall + Vars[:, [i]]
            x1 = x1 + self.cp_matrix[i] * Vars[:, [i]]
            x2 = x2 + self.ef_matrix[i] * Vars[:, [i]]
        x3 = [] # tp_energy
        for ind in range(Vars.shape[0]):
            select_nodes = np.where(Vars[ind]!=0)[0].tolist()
            _,tp_energy = get_center_node(select_nodes,self.tp_matrix)
            x3.append(tp_energy)
        x3 = np.vstack(x3)
        x_e = x3+x1
        pop.ObjV = - x_e + (x2**(1-self.b_ratio))/(1-self.b_ratio) # 计算目标函数值，赋值给pop种群对象的ObjV属性
        # 采用可行性法则处理约束
        pop.CV = np.hstack([np.abs(xall - self.select_num)])


def NodeChoose(cp_matrix,tp_matrix,ef_matrix,node_num,select_num,b_ratio):
    """================================实例化问题对象==========================="""

    problem = MyProblem(cp_matrix,tp_matrix,ef_matrix,node_num,select_num,b_ratio) # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'RI'       # 编码方式
    NIND = 100            # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.soea_DE_rand_1_L_templet(problem, population) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 500 # 最大进化代数
    myAlgorithm.mutOper.F = 0.5 # 差分进化中的参数F 0.5
    myAlgorithm.recOper.XOVR = 0.9 # 重组概率 0.7
    myAlgorithm.drawing = 0 # 不画图
    """===========================调用时间======================="""
    start = time.time()
    """===========================调用算法模板进行种群进化======================="""
    [best_p,population] = myAlgorithm.run() # 执行算法模板
    best_p.save() # 把最后一代种群的信息保存到文件中
    """===========================输出结果======================="""
    end = time.time()
    print ('Time {:.5f}'.format(end-start)) # 记录算法运算的时间
    # best_gen = np.argmin(problem.maxormins * obj_trace[:, 1]) # 记录最优种群个体是在哪一代
    best_ObjV = best_p.ObjV
    print('最优的目标函数值为：%s'%(best_ObjV))
    print('最优的决策变量值为：%s'%(best_p.Phen))
    select_nodes = np.where(best_p.Phen[0]==1)
    return select_nodes[0]

if __name__ == '__main__':
    select_nodes = NodeChoose(cp_matrix,tp_matrix,ef_matrix,node_num,select_num,b_ratio)
    print(select_nodes.tolist())

