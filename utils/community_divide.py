#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: jaculamata 
@license: Apache Licence  
@contact: 819436557@qq.com 
@site: http://blog.csdn.net/hqzxsc2006 
@software: PyCharm 
@file: 3_sim_formula.py 
@time: 2019/1/20 15:39 
"""

import pandas as pd
import time
import math
from copy import deepcopy
from data_process import data_process
import gc
import numpy as np
import random
from protect_comloc import protect_comloc


class community_divide():
    def __init__(self):
        pass

    # path="G:/pyfile/relation_protect/src/data/city_data/"
    def set_checkins(self, path, city):
        data_processor = data_process()
        data_processor.set_basic_info(path, city)
        data_processor.user_pairs()
        self.pairs_path = data_processor.pairs_path
        self.path = data_processor.path
        self.city = data_processor.city
        users = data_processor.checkins.uid.unique().tolist()
        users.sort(reverse=False)
        self.users = np.array(deepcopy(users))  # 将用户的id进行从小到大的排序
        del users
        gc.collect()
        self.checkins = data_processor.checkins
        print("城市用户数为：", len(self.users))

    def run(self, m):
        community_divide_count = pd.DataFrame(self.user_cluster_list["clusterId"].value_counts()).reset_index()
        community_divide_count.columns = ['clusterId', 'nums']
        community_divide_count = community_divide_count[community_divide_count.nums != m].reset_index(drop=True)
        community_divide_count_LACK = community_divide_count[community_divide_count.nums < m].reset_index(drop=True)
        if (len(community_divide_count) == 1 and len(community_divide_count_LACK) == 1) or len(community_divide_count) == 0:
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
        # print(community_divide_count)

        # 进行划分，将社区内用户数大于l的划分成几个子列表
        community_divide_count_numover = community_divide_count[community_divide_count.nums > m]
        for row in community_divide_count_numover.itertuples(index=False, name=False):  # 社区中用户个数大于l的社区进行均匀划分
            uids = list(self.user_cluster_list[self.user_cluster_list.clusterId == row[0]].uid.values)
            nums = int(math.floor(len(uids) / m))  # 可以划分的个数
            i_list = list(map(lambda x: x * m, range(nums)))
            for i in i_list:
                uids_temp = uids[i:(i + m)]
                self.clusterId += 1
                for u in uids_temp:
                    self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
                del uids_temp[-len(uids_temp):0]
        return False

    def add_clusterid(self, uids):
        for u in uids:
            self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
        self.clusterId += 1

    def community_disturb(self, checkins, method, t):   # 将用户的轨迹合并，然后均分，能够严格保证社区内的用户特征完全一致
        ano_checkins = pd.DataFrame()
        uids = list(checkins.uid.unique())  # uid的降序排列
        uids.sort()
        # print(uids)
        checkin1 = checkins.groupby(by=['locid']).size().reset_index(name="locid_time")
        locids = [row[0] for row in checkin1.itertuples(index=False, name=False)]
        loc_times = [row[1] for row in checkin1.itertuples(index=False, name=False)]
        del checkin1  # 释放checkin1的内存空间
        gc.collect()
        # print(loc_times)
        checkins = checkins.sort_values(by=['uid'], ascending=True).reset_index(drop=True)  # uid的升序排列
        if len(uids) < t:    # 最后一个社区需要添加虚假用户
            lack_user_nums = t - len(uids)
            for i in range(lack_user_nums):
                uids.append(random.randint(5000000, 6000000))  # 随机产生lack_user_nums个用户
            uids.sort()
        for i in range(len(loc_times)):
            loc_time, locid = loc_times[i], locids[i]
            loc_checkins = checkins[checkins.locid == locid]
            if loc_time % t == 0:   # 如果签到次数能够均匀分配
                nums = int(len(loc_checkins) / t)
            else:  # 签到次数不能均匀分配，则向上取整
                nums = int(math.ceil(len(loc_checkins) / t))  # 每个用户理想的记录签到数
                request_nums = nums * t - loc_time  # 需要增加的签到记录个数
                request_checkins = pd.DataFrame()
                if request_nums < len(loc_checkins):  # 需要增加的签到记录数小于原有的签到记录，则直接选取
                    request_checkins = loc_checkins[(len(loc_checkins) - request_nums):len(loc_checkins)]
                else:
                    tail_checkin = loc_checkins.tail(1)
                    for i in range(request_nums):
                        request_checkins = pd.concat([request_checkins, tail_checkin], ignore_index=True)
                loc_checkins = pd.concat([loc_checkins, request_checkins], ignore_index=True)
                loc_checkins.reset_index(drop=True)
            for i in range(t):
                checkins_temp = deepcopy(loc_checkins[(nums*i):(nums*(i+1))])
                checkins_temp.loc[:, 'uid'] = uids[i]
                ano_checkins = pd.concat([ano_checkins, checkins_temp], ignore_index=True)
        ano_checkins.reset_index(drop=True)
        # print(ano_checkins)
        ano_checkins.to_csv("G:/pyfile/relation_protect/src/data/result_data/" + method + "/" + self.city + "_" + method + "_" + str(t) +
                            "_user_simple_community.data", index=False, sep='\t', header=False, mode='a')

    def community_divide_core(self, m, n, method, k=-1):
        from cal_similarity import cal_similarity

        self.clusterId = 0  # 初始聚类号
        self.checkins.loc[:, 'clusterid'] = -1  # 初始化社区id,-1表示该用户还没有划分到具体社区
        user_cluster_list = []  # 用于记录每个用户可能属于的聚类，格式为[user,clusterId]
        for user in self.users:
             tmp = [user, -1]
             user_cluster_list.append(tmp)
        self.user_cluster_list = pd.DataFrame(user_cluster_list, columns=["uid", "clusterId"])
        t = math.ceil(1 / n)  # 用户隐私需求社区中至少包含t个用户
        visitedusers = []     # 记录已经划分社区的用户
        satisfied_users = []  # 记录满足l邻居的社区中所有的用户

        print("第一步: 获取需要使用的相似度特征值")
        # if method == "comloc":  # 共同访问位置
        #     comloc_similarity = comloc_similarity()
        #     comloc_similarity.set_checkins(self.checkins, self.city)
        #     self.u1_u2_sim = comloc_similarity.cal_user_pairs()
        #     self.u1_u2_sim.to_csv(self.pairs_path + self.city + "_comloc1.similarity", index=False, header=False)
        # elif method == "freqloc":  # 频繁访问位置
        #     cal_similarity = cal_similarity()
        #     cal_similarity.set_checkins(self.checkins, self.city)
        #     self.u1_u2_sim = cal_similarity.cal_freqloc_user_pairs(k)
        #     self.u1_u2_sim.to_csv(self.pairs_path + self.city + "_freqloc.similarity", index=False, header=False)
        # else:                      # 位置访问频率分布
        #     cal_similarity = cal_similarity()
        #     cal_similarity.set_checkins(self.checkins.values.tolist(), self.city, 20, [29.816691, -95.456244, 29.679229, -95.286390])
        #     self.u1_u2_sim = cal_similarity.cal_user_pairs()
        #     self.u1_u2_sim.to_csv(self.pairs_path + self.city + "_freqscatter.similarity", index=False, header=False)

        self.u1_u2_sim = pd.read_csv(self.pairs_path + self.city + "_" + method + ".similarity", names=["u1", "u2", "similarity"], header=None)

        print("第二步：开始计算每个用户属于的聚类")
        # 2.计算每个用户属于的聚类,m为内部参数用来判定用户的邻居 ,用户隐私需求参数 n
        for user in self.users:
            if user not in visitedusers:  # 如果用户还没有被访问过，也就是没有划分社区的用户
                user_k_users = self.u1_u2_sim[((self.u1_u2_sim.u1 == user) | (self.u1_u2_sim.u2 == user))]  # 获得与user相关的所有用户对
                user_k_users = user_k_users.sort_values(by=["similarity"], ascending=False).reset_index(drop=True)  # 按照亲密度进行降序排序
                user_sim_users = user_k_users[user_k_users.similarity >= m]  # 选取亲密度>=m的用户对记录
                uids = set.union(set(user_sim_users['u1'].values), set(user_sim_users['u2'].values))  # 当前社区内的所有用户
                if len(uids) == 0:
                    uids.add(user)
                uids = list(uids - set(visitedusers))
                if t < len(uids):  # 如果社区内用户数大于t个，则对社区进行分裂
                    nums = int(math.floor(len(uids) / t))  # 可以划分的社区的个数，比如用户有10个，社区用户数为3时，能划分3个满足条件的社区和1个用户不足的社区，这里得到的值是可满足条件的社区个数
                    i_list = list(map(lambda x: x * t, range(nums)))  # 构造一个便于取数的数组，比如起始值为[0,3,6]
                    for i in i_list:
                        uids_temp = deepcopy(uids[i:(i + t)])
                        for u in uids_temp:
                            self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
                            self.checkins.loc[self.checkins.uid == u, 'clusterid'] = self.clusterId
                        self.clusterId += 1
                        satisfied_users.extend(uids_temp)    # 将用户添加到满足条件的用户中
                        del uids_temp[-len(uids_temp):0]
                    uids_left = uids[(t * nums):len(uids)]   # 剩余个数没有t个的用户组成一个社区块
                    self.add_clusterid(uids_left)
                elif t == len(uids):
                    for u in uids:
                        self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
                        self.checkins.loc[self.checkins.uid == u, 'clusterid'] = self.clusterId
                    self.clusterId += 1
                    satisfied_users.extend(uids)  # 将用户添加到满足条件的用户中
                else:
                    self.add_clusterid(uids)     # 邻居用户个数不满足t个的作为一个社区
                visitedusers.extend(uids)
        # print(len(visitedusers))
        self.user_cluster_list.to_csv("G:/pyfile/relation_protect/src/data/city_data/" + self.city + "_" + method + "_" + str(t) + "_user_cluster_list", sep='\t',
            header=False, index=False)

        # 释放不需要使用的内存
        del self.u1_u2_sim
        gc.collect()
        print("第三步: 将社区内合数不满足l个的社区进行合并")

        # 得到所有的不满足条件的用户
        # print(len(satisfied_users))
        unsatified_users = list(set(self.users) - set(satisfied_users))
        # print(len(unsatified_users))
        user_cluster_list = pd.DataFrame()
        for u in unsatified_users:
            user_cluster_list = pd.concat([user_cluster_list, self.user_cluster_list[self.user_cluster_list.uid == u]], ignore_index=True)
        user_cluster_list.reset_index(drop=True)
        self.user_cluster_list = user_cluster_list
        while True:
            flag = self.run(t)
            if flag is True:
                break
        checkins = deepcopy(self.checkins[self.checkins.clusterid == -1])  # 还没有进行划分社区的用户记录
        self.checkins = self.checkins[~self.checkins.clusterid.isin([-1])]  # 先将没有划分社区的用户记录删除
        self.checkins.to_csv("G:/pyfile/relation_protect/src/data/result_data/" + method + "/" + self.city + "_" + method + "_" + str(t) +
                            "_user_simple_community.data", index=False, sep='\t', header=False)
        for row in self.user_cluster_list.itertuples(index=False, name=False):   # 对用户进行社区划分
            checkins.loc[checkins.uid == row[0], 'clusterid'] = row[1]

        # 第四步完成对原始数据的社区划分
        print("第四步: 完成对原始数据的社区划分")
        community_checkins = checkins.groupby(["clusterid"])  # 将原始所有的不满足条件的用户根据社区进行分组
        # print(len(community_checkins))
        for group in community_checkins:       # 分组内将用户的轨迹合并，然后均分，能够严格保证社区内的用户特征完全一致
            self.community_disturb(group[1], method, t)

        #  读取社区划分之后的数据
        self.checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/" + method + "/" + self.city + "_" + method + "_" + str(t) + "_user_simple_community.data",
                                    delimiter="\t", index_col=None,
                                    names=['uid', 'times', 'latitude', 'longitude', 'locid', 'clusterid'])
        self.checkins = self.checkins.sort_values(by=['uid'], ascending=True).reset_index(drop=True)
        # print(self.checkins)
        # print(len(self.checkins.uid.unique()))
        print("第五步: 进行特征扰动")
        # if method == "comloc":  # 共同访问位置
        #     pc = protect_comloc(t, method)
        #     pc.comnunity_disturb(self.checkins, 3, 0.2)
        # elif method == "freqloc":  # 频繁访问位置
        #    pass
        # else:  # 位置访问频率分布
        #    pass

        print("第六步: 进行可用性与安全性的计算")


if __name__ == "__main__":
    start = time.time()
    for i in [0.35, 0.3, 0.2, 0.19, 0.15, 0.13, 0.12, 0.1]:
        start1 = time.time()
        test_case = community_divide()
        test_case.set_checkins("G:/pyfile/relation_protect/src/data/city_data/", "1")
        test_case.community_divide_core(0.54, i, "comloc")
        print(str(time.time()-start1))
    end = time.time()
    print(str(end-start))
    pass  