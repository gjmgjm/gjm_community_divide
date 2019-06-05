#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: jaculamata 
@license: Apache Licence  
@contact: 819436557@qq.com 
@site: http://blog.csdn.net/hqzxsc2006 
@software: PyCharm 
@file: test.py 
@time: 2019/1/20 16:30 
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel,delayed
import math
import copy
from utility import utility
from community_divide import community_divide
from security_unility import security_unility
from collections import Counter
from itertools import combinations
from cal_similarity import cal_similarity
from community_divide4 import community_divide2
from sys import  argv
def test1():
    # print "test"
    print("test")
    pass


city = 0
u1_list = [1, 1, 3, 1, 0]
u2_list = [0, 2, 1,0,1]
u3_list = [1,1,1,0,0]

u4_list = [20,32,88,102,112,114,124,162,163,166,172,178,209,224,236,242,268,278,290,302,375,436,439]
import time
from numpy.linalg import norm
from scipy.stats import entropy
from joblib import Parallel, delayed
import multiprocessing
from scipy.special import comb, perm
import time
import gc
from copy import deepcopy
from community_divide4 import community_divide2

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M, base=2) + entropy(_Q, _M, base=2))
def single_security(u1, u2, checkins, checkins_obf):

    u1_checkin = checkins.loc[checkins.uid == u1]
    u1_checkin_obf = checkins_obf.loc[checkins_obf.uid == u1]
    u2_checkin = checkins.loc[checkins.uid == u2]
    u2_checkin_obf = checkins_obf.loc[checkins_obf.uid == u2]

    u1_locids = u1_checkin.locid.unique()
    u2_locids = u2_checkin.locid.unique()
    u1_locids_obf = u1_checkin_obf.locid.unique()
    u2_locids_obf = u2_checkin_obf.locid.unique()

    u1_u2_com_locids = set(u1_locids).intersection(set(u2_locids))
    u1_u2_com_locids_obf = set(u1_locids_obf).intersection(set(u2_locids_obf))
    if len(u1_u2_com_locids_obf) == 0:
        if len(u1_u2_com_locids) == 0:
            return 1
        else:
            return 0
    u1_u2_com = list(u1_u2_com_locids.intersection(u1_u2_com_locids_obf))
    print(u1, u2, len(u1_u2_com) * 1.0 / len(u1_u2_com_locids_obf))
    return len(u1_u2_com) * 1.0 / len(u1_u2_com_locids_obf)
def a_security( checkins, checkins_obf):
    start = time.time()
    print("用户间共同访问位置改变情况")
    pairs = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/1" + ".pairs", names=["u1", "u2"], header=None)
    checkins_obf_uids = checkins_obf.uid.unique()
    core_num = multiprocessing.cpu_count()
    # meet_cell = Parallel(n_jobs=core_num)(delayed(self.single_security)(pairs.iloc[i]['u1'], pairs.iloc[i]['u2']) for i in range(len(pairs)))
    meet_cell = Parallel(n_jobs=core_num)(
            delayed(single_security)(pairs.iloc[i]['u1'], pairs.iloc[i]['u2'], checkins, checkins_obf) for i in range(len(pairs)))
    print(sum(meet_cell))
    security = sum(meet_cell)/(comb(len(checkins_obf_uids), 2))
    end = time.time()
    print("总共花的费时间为", str(end - start))
    print("a:", security)
    return security
def testaaaa():
    # 随机生成两个离散型分布
    x = [np.random.randint(1, 11) for i in range(10)]
    print(x)
    print(np.sum(x))
    px = x / np.sum(x)
    print(np.array(x))
    print(px)
    y = [np.random.randint(1, 11) for i in range(10)]
    print(y)
    print(np.sum(y))
    py = y / np.sum(y)
    print(py)

    # 利用scipy API进行计算
    # scipy计算函数可以处理非归一化情况，因此这里使用
    # scipy.stats.entropy(x, y)或scipy.stats.entropy(px, py)均可
    KL = entropy(x, y)
    print(KL)




list1 = [1, 2, 3, 4, 5, 6, 7]
list2 = [7, 6, 5, 4, 3, 3, 2]
list3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list4 = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 9, 10, 11, 12, 13, 14, 15]

def get(a):
    if a>0:
        return True
    return False


class test():

    def __init__(self, k):
        # self.u1_u2_sim = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/" + "1_comloc.similarity",
        #                         names=["u1", "u2", "similarity"], header=None)
        # self.user_list = set(self.u1_u2_sim.u1.unique()).union(set(self.u1_u2_sim.u2.unique()))
        if get(k) is True:
            print("true")
        else:
            print("FLASE")

        # checkins = checkins[checkins.clusterid.isin([9, 19, 22, 31, 34, 38, 41, 43, 46, 47, 77, 103, 107, 122, 124, 125, 151, 153, 154, 155, 162, 172, 182, 204, 217, 267, 417, 534, 535, 569, 570, 574])]
        # checkins = checkins.sort_values(by=['clusterid', 'uid'], ascending=False).reset_index(drop=True)
        # print(checkins)
        # data = self.u1_u2_sim.values.tolist()
        # print(type(self.u1_u2_sim))
        # print(type(data))
        # print(len(self.user_list))
    def set(self):
        self.checkins_obf = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/1.csv",
                                        sep='\t',index_col=False)
    def check(self, method, city, l):
        checkins_divided = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/" + method + "/" + city + "_" + method + "_" + str(l) + "_user_simple_community.checkins", sep='\t',names=['uid', 'time', 'latitude', 'longitude', 'locid','clusterid'])
        checkins_divided = checkins_divided.sort_values(by=['clusterid', 'uid'], ascending=True).reset_index(drop=True)
        # checkins = checkins_divided[checkins_divided.]
        clusterList = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/1_" + method + "_" + str(l) + "_satisfied_clusterId.csv",index_col=False, names=['clusterId', 'uids_nums', 'p_nums'] )
        clusterList_lack = clusterList[clusterList.p_nums < l]
        print(len(clusterList.clusterId.unique()), len(clusterList_lack), clusterList_lack.uids_nums.sum())
    def pri(self,checkins):
        cluster_checkins = pd.DataFrame(columns=['A', 'B', 'C', 'D'])
        print(len(checkins))
        print(checkins)
        print()
        # return np.array(checkins)
        checkins.to_csv("G:/pyfile/relation_protect/src/data/city_data/result"+str(checkins.iloc[0, 3])+".csv", sep='\t',header=None,index=False)
    def test1(self, user_list):
         u1_u2_sim = self.u1_u2_sim
         for u1 in user_list:
             sim = 0.0
             for u2 in user_list:
                if u2 != u1:
                    temp = copy.deepcopy(u1_u2_sim[((u1_u2_sim.u1 == u1)&(u1_u2_sim.u2 == u2))|((u1_u2_sim.u1 == u2)&(u1_u2_sim.u2 == u1))])
                    sim += temp['similarity'].values[0]
             sim = sim/(len(user_list)-1)
             print(sim)
    def test2(self):


        # 随机生成两个离散型分布
        x = [np.random.randint(1, 11) for i in range(10)]
        print(x)
        print(np.sum(x))
        px = x / np.sum(x)
        print(px)
        y = [np.random.randint(1, 11) for i in range(10)]
        print(y)
        print(np.sum(y))
        py = y / np.sum(y)
        print(py)

        # 利用scipy API进行计算
        # scipy计算函数可以处理非归一化情况，因此这里使用
        # scipy.stats.entropy(x, y)或scipy.stats.entropy(px, py)均可
        KL = entropy(x, y)
        print(KL)
    def avg_similarity(self, u1, user_list):
        u1_u2_sim = self.u1_u2_sim
        user_list1 = copy.deepcopy(user_list)
        user_list1.remove(u1)
        sim = 0.0
        for u2 in user_list1:
            temp = u1_u2_sim[((u1_u2_sim.u1 == u1) & (u1_u2_sim.u2 == u2)) | ((u1_u2_sim.u1 == u2) & (u1_u2_sim.u2 == u1))]
            sim += temp['similarity'].values[0]
        sim = sim/len(user_list1)
        return [u1, sim]
    def test3(self, user_list):
        core_num = multiprocessing.cpu_count()
        user_cluster_avgsim = Parallel(n_jobs=core_num)(delayed(self.avg_similarity)(u, user_list) for u in user_list)
        user_cluster_avgsimlist = pd.DataFrame(user_cluster_avgsim, columns=['uid', 'avg_sim'])
        for uid in user_list:
            index = user_cluster_avgsimlist[user_cluster_avgsimlist.uid == uid].avg_sim.values[0]
            print(index)
    def test4(self):
        data = pd.DataFrame({'A': u1_list,
                             'B': u2_list,
                             # 'C': [u1_list,u1_list,u1_list,u1_list,u1_list],
                             'D': u3_list})
        print(data)
        # u1_freq_locids = pd.DataFrame(data, columns=['A', 'D'])  # 统计用户u1的locid及访问次数
        u1_freq_locids = data.groupby(by=['A', 'D']).size().reset_index(name='times')
        u1_freq_locids = u1_freq_locids.sort_values(by=['times'], ascending=False).reset_index(drop=True)

        print(u1_freq_locids)
        print(data)
        data.to_csv("G:/pyfile/relation_protect/src/data/test.csv", sep='\t', index=False, header=False)
        data1 = pd.read_csv("G:/pyfile/relation_protect/src/data/test.csv", sep='\t', names=['a', 'b', 'c'])
        print(data1)
        # row = data[0:1].values.tolist()[0]
        # print(row)
        # list5 = [1]
        # dataq = pd.DataFrame()
        # for i in list5:
        #     dataq = pd.concat([dataq, data[data.A == i]], ignore_index=True)
        # dataq.reset_index(drop=True)
        # print(data)
        # data = copy.deepcopy(dataq)
        #
        # print(data)
        # print(dataq)

        # data1 = copy.deepcopy(data[0:3])   # 引用
        # print(data)
        # print(data1)
        # data['B'] = data['B'].apply(lambda x: 0)
        # for i in range(len(data1)):
        #    data1.iloc[i, 1] = 0
        # print(data)
        # print(data1)

        # extra_uid_list = list((set.union(set(data.A.unique()), set(data.B.unique()))))
        # extra_uid_list.remove(0)
        # print(extra_uid_list)
        # print(data)
        # community_divide_count = pd.DataFrame(data["A"].value_counts()).reset_index()
        # community_divide_count.columns = ['A', 'nums']
        # print(community_divide_count)
        # print(data)
        # community_divide_count.iloc[0, 0] = 'B'
        # print(community_divide_count)
        # print(data)
    def test(self, m, n):
        num = 0
        n = math.ceil(1/n)
        for user in self.user_list:
            user_k_users = self.u1_u2_sim[((self.u1_u2_sim.u1 == user) | (self.u1_u2_sim.u2 == user))]
            user_k_users = user_k_users.sort_values(by=["similarity"], ascending=False).reset_index(drop=True)  # 按照亲密度进行排序
            user_sim_users = user_k_users[user_k_users.similarity >= m]
            if len(user_sim_users) + 1 < n:
                num += 1
        f = open('G:/pyfile/relation_protect/src/data/city_data/log.txt', 'a', encoding='UTF-8')
        f.write(str(m)+'需要添加虚假用户的user个数为：' + str(n)+' '+str(num)+'\n')
        f.close()
        print("需要添加虚假用户的user个数为：", n, num)
        # data = pd.DataFrame({'A': u1_list,
        #                      'B': u2_list,
        #                      'C': [u1_list,u1_list,u1_list,u1_list,u1_list],
        #                      'D': u3_list})
        # print(data)
        # community = data.groupby("D")
        # print(len(community))
        # core_num = multiprocessing.cpu_count()
        # Parallel(n_jobs=core_num)(delayed(self.pri)(group[1]) for group in community)
        # # pd.DataFrame(checkins).to_csv("G:/pyfile/relation_protect/src/data/city_data/result.csv", sep='\t', header=None, index=False)

        # user1 = "1"
        # user2 = "2"
        # if user1 != user2:
        #     print("11111")
        # user = 1
        # user_list = list((set.union(set(data['A'].values), set(data['B'].values))))
        # print(user_list)
        # user1 = user_list
        # user1.remove(user)
        # print(user_list)
        # print(user1)
        # union_loc = set.union(set(data['A'].values), set(data['B'].values))
        # C = data.iloc[1].C
        # print(type(C))
        # print(C)
        # b = np.random.choice(C)
        # print(b)
        # data.iloc[1,1] = b
        # print(data.iloc[1,1])
        # for index, user in data:
        #     print(index,user)
        # union_loc = set(data.A.unique()).union(set(data.B.unique()))
        # list1 = list(set(data['A'].values).intersection(set(data['B'].values)))
        # list1 = [ 0 for i in range(len(data['B']))]
        # data['B'] = list1
        # data['B'] = data['B'].apply(lambda x: 0)
        # user_list = list(set(data.A + data.B))
        # print(data.B.unique()[0])
        # print(data)
    def run(self):

        community_divide_count = pd.DataFrame(self.user_cluster_list["clusterId"].value_counts()).reset_index()
        community_divide_count.columns = ['clusterId', 'nums']
        # print(community_divide_count)
        community_divide_count = community_divide_count[community_divide_count.nums != 4].reset_index(drop=True)
        print(community_divide_count)
        community_divide_count_LACK = community_divide_count[community_divide_count.nums < 4].reset_index(drop=True)

        if (len(community_divide_count) == 1 and len(community_divide_count_LACK) == 1) or len(community_divide_count)==0:
            return True

        # 进行首尾合并
        j = len(community_divide_count)
        for i in range(len(community_divide_count)):
            j = j - 1
            header_clusterId = community_divide_count.loc[i, 'clusterId']
            if i > j:
                break
            tail_clusterid = community_divide_count.loc[j, 'clusterId']
            uids = list(self.user_cluster_list[self.user_cluster_list.clusterId == tail_clusterid].uid.values)
            for u in uids:
                self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = header_clusterId

        community_divide_count = pd.DataFrame(self.user_cluster_list["clusterId"].value_counts()).reset_index()
        community_divide_count.columns = ['clusterId', 'nums']
        print(community_divide_count)

        # 进行划分，将社区内用户数大于l的划分成几个子列表
        community_divide_count_numover = community_divide_count[community_divide_count.nums > 4]
        for row in community_divide_count_numover.itertuples(index=False, name=False):  # 社区中用户个数大于l的社区进行均匀划分
            uids = list(self.user_cluster_list[self.user_cluster_list.clusterId == row[0]].uid.values)
            nums = int(math.floor(len(uids) / 4))  # 可以划分的个数
            i_list = list(map(lambda x: x * 4, range(nums)))
            for i in i_list:
                uids_temp = uids[i:(i + 4)]
                self.clusterId += 1
                for u in uids_temp:
                    self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
                del uids_temp[-len(uids_temp):0]

        return False
    def community_disturb(self, checkins, t):  # 将用户的轨迹合并，然后均分，能够严格保证社区内的用户特征完全一致,t是社区内用户数
        ano_checkins = pd.DataFrame()
        uids = list(checkins.uid.unique())  # uid的降序排列
        uids.sort()
        checkin1 = checkins.groupby(by=['locid']).size().reset_index(name="locid_time")
        locids = [row[0] for row in checkin1.itertuples(index=False, name=False)]
        loc_times = [row[1] for row in checkin1.itertuples(index=False, name=False)]
        del checkin1  # 释放checkin1的内存空间
        gc.collect()
        checkins = checkins.sort_values(by=['uid'], ascending=True).reset_index(drop=True)  # uid的降序排列
        for i in range(len(loc_times)):
            loc_time, locid = loc_times[i], locids[i]
            loc_checkins = checkins[checkins.locid == locid]
            if loc_time % t == 0:  # 如果签到次数能够均匀分配
                nums = int(len(loc_checkins) / t)
            else:  # 签到次数不能均匀分配，则向上取整
                nums = int(math.ceil(len(loc_checkins) / t))  # 每个用户理想的记录签到数
                request_nums = nums * t - loc_time  # 需要增加的签到记录个数
                request_checkins = pd.DataFrame()
                if request_nums < len(loc_checkins):   # 需要增加的签到记录数小于原有的签到记录，则直接选取
                    request_checkins = loc_checkins[(len(loc_checkins)-request_nums):len(loc_checkins)]
                else:
                    tail_checkin = loc_checkins.tail(1)
                    for i in range(request_nums):
                        request_checkins = pd.concat([request_checkins, tail_checkin], ignore_index=True)
                loc_checkins = pd.concat([loc_checkins, request_checkins], ignore_index=True)
                loc_checkins.reset_index(drop=True)
            for i in range(t):
                checkins_temp = deepcopy(loc_checkins[(nums * i):(nums * (i + 1))])
                checkins_temp.loc[:, 'uid'] = uids[i]
                ano_checkins = pd.concat([ano_checkins, checkins_temp], ignore_index=True)
        ano_checkins.reset_index(drop=True)
        ano_checkins.to_csv("G:/pyfile/relation_protect/src/data/city_data/test2.csv", sep='\t', header=False, index=False)
    def test_divide(self):
        import gc
        list3 = list(range(42))  # 用户id
        dicta = {'uid': list3, 'clusterId': list4}
        # user_cluster_list = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/1_freqscatter_3_user_cluster_list", sep='\t', header=None, names=["uid", "clusterIds", "clusterId", "cluster_sims"])
        user_cluster_list = pd.DataFrame(dicta, columns=['uid', 'clusterId'])
        self.user_cluster_list = copy.deepcopy(user_cluster_list).ix[:, [0, 2]]
        del user_cluster_list
        gc.collect()
        self.clusterId = 1070
        while True:
            flag = self.run()
            if flag is True:
                break
        community_divide_count = pd.DataFrame(self.user_cluster_list["clusterId"].value_counts()).reset_index()
        community_divide_count.columns = ['clusterId', 'nums']
        print(community_divide_count)
        community_divide_count = community_divide_count[community_divide_count.nums != 4].reset_index(drop=True)
        print(community_divide_count)
        print(self.user_cluster_list)
        del self.user_cluster_list
        gc.collect()
    def test_divide1(self):
        checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/test.csv", delimiter="\t", index_col=None)
        self.community_disturb(checkins, 3)
    def aaa(self,checkins):
        uids = list(checkins.uid.unique())
        checkin_obf = self.checkins_obf[self.checkins_obf.uid.isin(uids)]
        ano_checkins = checkin_obf[91:92]
        for u in uids:
            u_checkin_obf = checkin_obf[checkin_obf.uid == u]
            locid_obf = pd.DataFrame(u_checkin_obf['locid'].value_counts()).reset_index()
            locid_obf.columns = ["locid", "cnt"]
            locid_obf = locid_obf.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
            u_checkins = checkins[checkins.uid == u]
            locid = pd.DataFrame(u_checkins['locid'].value_counts()).reset_index()
            locid.columns = ["locid", "cnt"]
            locid = locid.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
            union = [list(locid_obf[0:3].locid.unique()), list(locid[0:3].locid.unique())]


start = [[0,2], [0,3], [4,6], [4,7], [7,8], [9,10]]



def recursion_user(resemble_list, user):
    temp = list(resemble_list[resemble_list.start == user].end.values)
    if len(temp) == 0:
        return temp
    else:
        temp1 = deepcopy(temp)
        for u in temp1:
            list_temp = recursion_user(resemble_list, u)
            temp.extend(list_temp)
    return temp


from math import cos, sin, atan2, sqrt, pi, radians, degrees


def center_geolocation(geolocations):
    x = 0
    y = 0
    z = 0
    lenth = len(geolocations)
    for lat, lon in geolocations:
        lon = radians(float(lon))
        lat = radians(float(lat))
        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)
    x = float(x / lenth)
    y = float(y / lenth)
    z = float(z / lenth)
    # return (degrees(atan2(y, x)), degrees(atan2(z, sqrt(x * x + y * y))))
    return degrees(atan2(z, sqrt(x * x + y * y))), degrees(atan2(y, x))


def cal_delta(x):
    circle_len = 2 * 6371.004 * math.pi
    circle_len_1 = circle_len * math.cos(x * math.pi / 180)
    delta_lng = 360 / circle_len_1
    delta_lat = 360 / (2 * 6371.004 * math.pi)
    return [delta_lat / 2, delta_lng / 2]
    # 两点之间的欧氏距离计算

from math import radians, cos, sin, asin, sqrt

def Haversine(lat1, lon1, lat2, lon2):
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # 地球平均半径，单位为公里
    d = c * r
    print(d)
    print("该两点间距离={0:0.3f} km".format(d))

def euclidean_distance(loc1, loc2, lons_per_km, lats_per_km):
    return math.sqrt(((loc1[1] - loc2[1]) / lons_per_km) ** 2 + ((loc1[0] - loc2[0]) /lats_per_km) ** 2)

import pyproj


def proj_trans(path, city):
   checkins = pd.read_csv(path + "city_data/" + city + ".csv", delimiter="\t", index_col=None)
   # data = pd.read_excel(u"D:/Visualization/python/file/location.xlsx")
   checkins.loc[:, 'timestamp'] = checkins['time'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))  # 将签到记录中的日期时间转换成时间戳
   checkins = checkins.sort_values(by=['timestamp'], ascending=False).reset_index(drop=True)
   lon = checkins.longitude.values
   lat = checkins.latitude.values
   p1 = pyproj.Proj(init="epsg:4326")  # 定义数据地理坐标系
   p2 = pyproj.Proj(init="epsg:3857")  # 定义转换投影坐标系
   x1, y1 = p1(lon, lat)
   x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
   checkins['lon'] = x2
   checkins['lat'] = y2
   i = 0
   file = open("G:/pyfile/relation_protect/src/data/" + "city_data/" + city + ".dat", 'a', encoding='UTF-8')
   for user in checkins.uid.unique():
       file.write('#'+str(i)+':'+'\n')
       file.write('>0:')
       u_lat_lon = checkins[checkins.uid == user].loc[:, ['lon', 'lat']]
       for row in u_lat_lon.itertuples(index=False, name=False):
           file.write(str(row[0])+','+str(row[1])+';')
       file.write('\n')
       i += 1
   file.close()


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M, base=2) + entropy(_Q, _M, base=2))

def insert(intervals, newInterval):
    if len(intervals) == 0 or intervals == None:
        if len(newInterval) == 0 or newInterval == None:
            return None
        else:
            return [newInterval]
    list1 = []
    index = 0
    result_list = []
    for i in range(len(intervals)):
        if intervals[i][1] < newInterval[0]:
            result_list.append(intervals[i])
        if intervals[i][1] >= newInterval[0]:
            list1.extend(intervals[i])
            index = i
            break
    j = 0
    for l in intervals[(index + 1):len(intervals)]:
        if newInterval[1] < l[0]:
            index += j
            break
        list1.extend(l)
        j += 1
    list1.extend(newInterval)
    result_list.append([min(list1), max(list1)])
    if index == -1 or index == len(intervals) - 1:
        return result_list
    else:
        for l in intervals[(index + 1):len(intervals)]:
            result_list.append(l)
    return result_list


if __name__ == "__main__":
    # insert([[1,2],[3,5],[6,7],[8,10],[12,16]],[4,8])
    print(insert([[1,5]],[0,0]))
    # js = JSD(np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0, 0]), np.array([3/17, 3/17, 3/17, 3/17, 2/17, 2/17, 1/17, 1/17]))
    # js = JSD(np.array([0.2, 0.4, 0.2, 0.1, 0.1, 0, 0]),
    #          np.array([2/14, 4/14, 0, 1/14, 1/14, 1/14, 1/14]))
    # print(js)
    # a = [1,2,3]
    # b = [4,5,6]
    # # am = [[1, 2], [1, 1], [1, 1]]
    # combine = [[x, y] for x in a for y in b]
    # print(combine)
    # am = pd.DataFrame(am,columns=['a', 'b'])
    # sum = am.b.sum()
    # am.loc[:,'b'] =  am.loc[:,'b']/sum
    # print(am)
    # sum = np.sum([x[1] for x in a])
    # a = [[line[0], float(line[1] / sum)] for line in a]  # 位置、频率
    # print(a)
    # for line in a:
    #     print(line[1])
    #     line[1] = float(math.exp(line[1]))
    #     print(type(line[1]))
    #     print(line)
    # print(math.exp(1))
    # print(a)
    # start = time.time()
    # argvs = argv
    # deta = float(argv[1])
    # t = float(argv[2])
    # l = int(argv[3])
    # print(deta,t,l)
    # start = time.time()
    # argv = [0.6, 0.6, 0.6, 3]
    # deta = math.exp(-1 / 3)
    # t = math.exp(-0.8)
    # l = argv[3]
    # for l in [3, 4, 5, 6, 7, 8, 9]:
    # for l in [7]:
    #     for i in [81, 82, 83, 84, 85]:
            # for a in [0.1, 0.2, 0.3, 0.4, 0.7, 0.8]:
            # for a in [0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54]:
            # for a in [0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.601, 0.602, 0.603, 0.604, 0.605, 0.609, 0.61, 0.62, 0.7, 0.75, 0.8]:
            # start1 = time.time()
            # test = community_divide2()
            # 奥斯汀
            # test.set_checkins("G:/pyfile/relation_protect/src/data/", "1", 30, 40, [30.387953, -97.843911, 30.249935, -97.635460])
            # test.community_divide_core(math.exp(-1 / 2), l, math.exp(-1 / 2), 0.5, 0.5, "comloc", 3, i)  # 500
            # test.community_divide_core(0.6, i, a, 0.5, 0.5, "comloc", 3)  # 500
            # test.community_divide_core(math.exp(-1/3), i, math.exp(-0.4), 0.5, 0.5, "comloc", 3)  # 400米
            # test.community_divide_core(math.exp(-0.5), i, math.exp(-1/3), 0.5, 0.5, "freqloc", 3)   # m,q
            # 旧金山
            # test.set_checkins("G:/pyfile/relation_protect/src/data/", "SF", 30, 40, [37.809524, -122.520352, 37.708991, -122.358712])
            # test.community_divide_core(math.exp(-1/3), i, math.exp(-0.4), 0.5, 0.5, "comloc", 3)  # 500
            # test.community_divide_core(math.exp(-1/3), i, math.exp(-0.4), 0.5, 0.5, "comloc", 3)  # 400米
            # test.community_divide_core(math.exp(-0.9), i, math.exp(-1/3), 0.5, 0.5, "freqloc", 3)  # m,q
            # SNAP NY
            # test.set_checkins("G:/pyfile/relation_protect/src/data/", "SNAP_NY_1", 40, 40, [40.836357, -74.052914, 40.656702, -73.875168])
            # test.community_divide_core(math.exp(-1/2), i, math.exp(-0.8), 0.5, 0.5, "comloc", 3)  # 500
            # test.community_divide_core(math.exp(-1/3), i, math.exp(-0.8), 0.5, 0.5, "comloc", 3)  # 500
            # test.community_divide_core(math.exp(-1/3), i, math.exp(-0.4), 0.5, 0.5, "comloc", 3)  # 400米
            # test.community_divide_core(math.exp(-0.5), i, math.exp(-1/4), 0.5, 0.5, "freqloc", 3)  # m,q
            # FS_NY
            #  扰动特征相似度 l多样性  发布特征相似度  多特征参数1、参数2  指定扰动特征  频繁访问位置特征参数
            # test.set_checkins("G:/pyfile/relation_protect/src/data/", "FS_NY_1", 40, 40, [40.836357, -74.052914, 40.656702, -73.875168])
            # test.community_divide_core(deta, l, t, 0.5, 0.5, "comloc", 3, i)  # 500
            # test.community_divide_core(math.exp(-1/3), i, math.exp(-0.8), 0.5, 0.5, "comloc", 3)  # 500
            # test.community_divide_core(math.exp(-1/3), i, math.exp(-0.4), 0.5, 0.5, "comloc", 3)  # 400米
            # test.community_divide_core(math.exp(-0.8), i, math.exp(-1/3), 0.5, 0.5, "freqloc", 3)  # m,q
            # print(str(time.time() - start1))
    # end = time.time()
    # print(str(end - start))
    # 113.270714,23.13552,114.064803,22.549054
    # -97.812884, -97.691613, 30.234639, 30.334892
    # [40.836357, -74.052914, 40.656702, -73.875168]
    # start = time.time()
    # locations = [[116.568627, 39.994879], [116.564791, 39.990511], [116.575012, 39.984311]]
    # locations = [[39.994879, 116.568627], [39.990511, 116.564791], [39.984311, 116.575012]]
    # print(center_geolocation(locations))
    # 30.2820555	-97.74402035  30.2679095833	-97.74931241670001
    # ins = cal_delta((30.2820555 + 30.2679095833)/2)
    # lons_per_km = ins[1] * 2
    # lats_per_km = ins[0] * 2    # -97.7520212333	30.2766266667
    # 30.274867466， -97.7516181167  30.2764877946 -97.7477324009  -97.7643307417	30.2754658
    # -97.7464980667	30.2780316667	-97.7477324009	30.2764877946  -97.75810565	30.2799416333
    # -73.94186034276925	40.823299028682534	-73.94619714398068	40.826860912891306
    # ins = cal_delta((40.823299028682534 + 40.826860912891306) / 2)
    # lons_per_km = ins[1] * 2
    # lats_per_km = ins[0] * 2
    # print(math.cos(((40.836357 + 40.656702)/2) * math.pi / 180))
    # print(ins)
    # print(euclidean_distance([30.2820555, -97.74402035], [30.2679095833, -97.74931241670001], lons_per_km, lats_per_km))
    # Haversine(30.2820555, -97.74402035, 30.2679095833, -97.74931241670001)
    # # print(euclidean_distance([40.836357, -74.052914], [40.656702, -73.875168], lons_per_km, lats_per_km))
    # Haversine(40.836357, -74.052914, 40.656702, -73.875168)
    # for k in [4, 5, 6, 7, 8, 9, 10]:
    # # for k in [3]:
    #     start = time.time()
    #     cal_similarity1 = cal_similarity("comloc")
    #     # checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/1.csv", delimiter="\t", index_col=None)
    #     # checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/comloc/1_disturb_accuracy_3_comloc.csv", sep='\t', names=["uid", "time", "latitude", "longitude", "locid", "clusterid"], header=None)
    #     # cal_similarity.set_checkins(checkins.values.tolist(), "1", 30, 40, [30.387953, -97.843911, 30.249935, -97.635460])
    #     # cal_similarity.cal_weighted_sim(4, 0.5, 0.5)
    #     checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/comloc/16_"+str(k)+"_comloc.csv",
    #                            sep='\t', names=["uid", "time", "latitude", "longitude", "locid", "grid_id", "clusterid", "is_core", "grid_id_before", "locid_before"],
    #                            header=None)
    #     checkins = checkins.sort_values(by=['uid'], ascending=False).reset_index(drop=True)
    #     checkins = checkins.ix[:, [0, 1, 2, 3, 4]]
    #     cal_similarity1.set_checkins(checkins.values.tolist(), "1", k, 30, 40, [30.387953, -97.843911, 30.249935, -97.635460])
    #     cal_similarity1.cal_comloc_user_pairs()
    #     cal_similarity1.cal_freqloc_user_pairs(3)
    #     end = time.time()
    #     print(str(end-start))

    # for i in [ ]:
    # for i in [3, 4, 5, 6, 7, 8, 9, 10]:
    #     start1 = time.time()
    #     test = community_divide2()
    #     test.set_checkins("G:/pyfile/relation_protect/src/data/city_data/", "1", 30, 40)
    #     test.community_divide_core(math.exp(-1 / 3), i, math.exp(-1 / 2), 0.5, 0.5, "comloc", 3)
    #     # test.community_divide_core(math.exp(-1 / 2), i, math.exp(-1/3), 0.5, 0.5, "freqloc", 3)
    #     print(str(time.time() - start1))
    # end = time.time()
    # print(str(end - start))
        # test1 = test(3)

        # uids = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11]
        # resemble_list = pd.DataFrame(start, columns=['start', 'end'])
        # simusers = []
        # temp_uids = deepcopy(uids)
        # for i in range(4):
        #     core_user = temp_uids[0]
        #     core_simusers = [core_user]
        #     core_simusers.extend(recursion_user(resemble_list, core_user))  # 跟核心用户相似的用户， 先对与核心用户相似的用户进行位置扰动
        #     simusers.append(core_simusers)
        #     for user in core_simusers:
        #         temp_uids.remove(user)
        #
        # print(simusers)
        # resemble_list.loc[:,'ended'] = resemble_list['end']
        # print(resemble_list)

        # uids = list(resemble_list.start.unique())
        # print(uids)
        # for k in [3]:
        #     start = time.time()
        #     cal_similarity1 = cal_similarity("comloc")
        #     checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/1.csv", delimiter="\t",
        #                            index_col=None)
        #     u1_freq_locids = pd.DataFrame(checkins[checkins.uid == 0].locid.value_counts()).reset_index().rename(columns={'locid':'cnt', 'index':'locid'})  # 统计locid的不同值及其个数
        #     # print(type(u1_freq_locids))
        #     u2_freq_locids = pd.DataFrame(checkins['locid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
        #     u2_freq_locids.columns = ['locid', 'cnt']
        #     print(u1_freq_locids)
        #     print(u2_freq_locids)
            # checkins = checkins.sort_values(by=['uid'], ascending=False).reset_index(drop=True)
            # cal_similarity1.set_checkins(checkins.values.tolist(), "1", k, 30, 40,
            #                              [30.387953, -97.843911, 30.249935, -97.635460])
            # # cal_similarity1.cal_freqloc_user_pairs(3)
            # cal_similarity1.cal_comloc_user_pairs()
            # end = time.time()
            # print(str(end - start))
        # test1.check("freqloc", "1", 3)
        # test1.set()
        # checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/comloc/12_3_comloc.csv",
        #     delimiter="\t", names=["uid", "time", "latitude", "longitude", "locid", "clusterid", "is_core"], header=None)
        # checkins = checkins[checkins.clusterid.isin([0, 2, 4, 6, 8, 13, 17, 18, 21, 24, 26, 27, 29, 30, 32, 41, 57, 65, 113, 234, 390, 394, 399, 401, 403, 404, 406, 413, 415, 416, 420, 421, 423, 565, 568, 570, 578, 580, 592, 593, 595, 598, 607, 613, 616, 628])]
        # com = checkins.groupby(by=['clusterid'])
        # for group in com:
        #     test1.aaa(group[1])
        # start = time.time()
        # for i in [4, 5, 6, 7, 8, 9, 10]:
        #     # for i in [3]:
        #     start1 = time.time()
        #     test = community_divide2()
        #     test.set_checkins("G:/pyfile/relation_protect/src/data/city_data/", "1", 30, 40)
        #     test.community_divide_core(0.54, i, math.exp(-1 / 2), 0.5, 0.5, "comloc", 3)
        #     # test.community_divide_core(0.50, i, 0.54, 0.5, 0.5, "freqloc", 3)
        #     print(str(time.time() - start1))
        # end = time.time()