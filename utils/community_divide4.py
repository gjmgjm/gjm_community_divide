#!/usr/bin/env python
# encoding: utf-8

import time
from data_process import data_process
from copy import deepcopy
import pandas as pd
import numpy as np
import gc
from disturb_comloc import disturb_comloc
from disturb_freqloc import disturb_freqloc
import math
from grid_divide import grid_divide
from community_security import commutity_security
from sys import argv

class community_divide2():

    def __init__(self):
        pass

    def set_checkins(self, path, city, n, m, range):
        data_processor = data_process()
        data_processor.set_basic_info(path, city)
        self.pairs_path = data_processor.pairs_path
        self.path = data_processor.path
        self.city = data_processor.city
        self.range = range

        users = data_processor.checkins.uid.unique().tolist()
        users.sort(reverse=False)
        self.users = np.array(deepcopy(users))  # 将用户的id进行从小到大的排序
        del users
        gc.collect()
        self.M = m    # 经度网格
        self.N = n    # 纬度网格
        grid_divider = grid_divide(data_processor.checkins.values.tolist(), self.N, self.M, self.range)
        self.checkins = grid_divider.divide_area_by_NN().ix[:, [0, 1, 2, 3, 4, 5]]  # 带有网格id
        self.latInterval = grid_divider.get_latInterval()
        self.lngInterval = grid_divider.get_lngInterval()
        # self.lons_per_km = 0.005202 * 2  # delta longitudes per kilo meter  0.005681
        # self.lons_per_km = 0.005681 * 2  # 旧金山
        # self.lats_per_km = 0.004492 * 2  # delta latitudes per kilo meter
        self.lons_per_km = 0.0059352 * 2  # NY
        self.lats_per_km = 0.0044966 * 2  # delta latitudes per kilo meter
        print("城市用户数为：", len(self.users))

    def cal_l_feature(self, uids, q, l):  # 方法与社区合并类似
        uids_temp = deepcopy(uids)
        uid_temp = [uids_temp[0]]
        uids_temp.remove(uids_temp[0])
        num = 1
        resemble_list = []
        while len(uids_temp) > 0:
            u2 = uids_temp[0]
            temp_num = 0
            for u1 in uid_temp:
                u2_temp = u2
                u1_temp = u1
                if u1 > u2:
                    u1, u2_temp = u2_temp, u1
                if self.u1_u2_weighted_sim.xs(u1).xs(u2_temp)[0] > q:  # 不满足l多样性
                    resemble_list.append([u1_temp, u2])
                    break
                temp_num += 1
            if len(uid_temp) == temp_num:
                num += 1
            uid_temp.append(u2)
            uids_temp.remove(uids_temp[0])
            if num == l + 1:  # 说明当前至少有不同的特征l个，则划分一个社区
                uids_temp.insert(0, u2)
                uid_temp.remove(u2)
                for u in uid_temp:
                    self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
                self.satisfied_clusterId.append([self.clusterId, len(uid_temp), num-1, uid_temp, resemble_list])
                self.clusterId += 1
                if len(uids_temp) > 0:
                    uid_temp = [uids_temp[0]]
                    self.user_cluster_list.loc[self.user_cluster_list.uid == uids_temp[0], 'is_core'] = 1
                    resemble_list = []
                    uids_temp.remove(uids_temp[0])
                    num = 1
        if len(uid_temp) != 0:  # 最后可能还是有没有l个特征的社区
            for u in uid_temp:
                self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
            self.satisfied_clusterId.append([self.clusterId, len(uid_temp), num, uid_temp, resemble_list])
            self.clusterId += 1

    def cal_community_l_feature(self, uids1, uids2, q, l, p_nums, resemblelist):
        """
        :param uids1: 首社区中用户
        :param uids2: 尾社区中用户
        :param q:    社区l多样性的判断条件，即相似度值
        :param l:    社区特征多样性个数l
        :param p_nums:  首社区中p特征的个数
        :return:
        依次将尾社区中的元素添加到首社区中，并判断p特征的个数，当p特征的个数不超过2*l时，进行一次社区分裂
        """
        uids_temp = deepcopy(uids2)
        uid_temp = deepcopy(uids1)
        num = p_nums
        resemble_list = deepcopy(resemblelist)
        while len(uids_temp) > 0:
            u2 = uids_temp[0]
            temp_num = 0
            for u1 in uid_temp:
                u2_temp = u2
                u1_temp = u1
                if u1 > u2:
                    u1, u2_temp = u2_temp, u1
                if self.u1_u2_weighted_sim.xs(u1).xs(u2_temp)[0] > q:  # 不满足l多样性
                    resemble_list.append([u1_temp, u2])
                    break
                temp_num += 1
            if len(uid_temp) == temp_num:
                num += 1
            uid_temp.append(u2)
            uids_temp.remove(uids_temp[0])
            if num == l+1:  # 说明当前至少有不同的特征l个，则划分一个社区
                uids_temp.insert(0, u2)
                uid_temp.remove(u2)
                for u in uid_temp:
                    self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
                # 记录社区划分的中间结果
                self.unsatisfied_clusterId_temp.append([self.clusterId, len(uid_temp), num-1, uid_temp, resemble_list])
                self.clusterId += 1
                if len(uids_temp) > 0:    # 进行社区分裂之后，尾社区还有元素，进行下一次合并
                    uid_temp = [uids_temp[0]]
                    self.user_cluster_list.loc[self.user_cluster_list.uid == uids_temp[0], 'is_core'] = 1
                    uids_temp.remove(uids_temp[0])
                    resemble_list = []
                    num = 1
        if len(uid_temp) != 0:  # 记录最后一个没有进行社区分裂的社区
            for u in uid_temp:
                self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
            self.unsatisfied_clusterId_temp.append([self.clusterId, len(uid_temp), num, uid_temp, resemble_list])
            self.clusterId += 1

    def run(self, q, l):
        """
        :param q:  社区l多样性的判断条件，即相似度值
        :param l:  社区特征多样性个数l
        :return:
        """
        community_divide_count = self.unsatisfied_clusterId[self.unsatisfied_clusterId.p_nums < l].reset_index(drop=True)
        if (len(community_divide_count) == 1) or len(community_divide_count) == 0:
            return True
        # 进行首尾合并
        j = len(community_divide_count)
        for i in range(len(community_divide_count)):
            j = j - 1
            if i > j:
                break
            if i == j:
                row = community_divide_count[i:(i+1)].values.tolist()[0]   # 社区个数为奇数，最后i和j相同，只剩一个社区
                self.unsatisfied_clusterId_temp.append(row)
                break
            header_clusterId = community_divide_count.loc[i, 'clusterId']
            tail_clusterid = community_divide_count.loc[j, 'clusterId']
            uids2 = community_divide_count[community_divide_count.clusterId == tail_clusterid].uids.values[0]
            uids1 = community_divide_count[community_divide_count.clusterId == header_clusterId].uids.values[0]
            uids2_core = self.user_cluster_list[(self.user_cluster_list.clusterId == tail_clusterid) & (self.user_cluster_list.is_core == 1)].uid.values[0]
            self.user_cluster_list.loc[self.user_cluster_list.uid == uids2_core, 'is_core'] = 0  # 将需要合并的尾社区的核心用户取消标记
            uids1_resemblelist = community_divide_count.loc[i, 'resemble_list']
            p_nums = self.unsatisfied_clusterId.loc[i, 'p_nums']
            self.cal_community_l_feature(uids1, uids2, q, l, p_nums, uids1_resemblelist)
        return False

    def run_sim(self, q, l, m):
        clusterlist = [row[0] for row in self.unsatisfied_clusterId.itertuples(index=False, name=False)]
        visited_cluster = []
        for i in range(len(self.unsatisfied_clusterId)):
            clusterID = self.unsatisfied_clusterId.loc[i, 'clusterId']
            uids1 = self.unsatisfied_clusterId.loc[i, 'uids']  # 具有特征多样性做多的一个社区
            p_nums = self.unsatisfied_clusterId.loc[i, 'p_nums']
            uids1_resemblelist = self.unsatisfied_clusterId.loc[i, 'resemble_list']
            core_user = uids1[0]  # 社区的核心用户
            all_uids2 = []
            if clusterID not in visited_cluster:  # 该社区没有被访问过
                visited_cluster.append(clusterID)
                clusterlist.remove(clusterID)
                visited_cluster_temp = []
                for clusterid in clusterlist:  # 与剩下的社区进行合并
                    num = 0
                    uids2 = self.unsatisfied_clusterId[self.unsatisfied_clusterId.clusterId == clusterid].uids.values[0]
                    core_user1 = uids2[0]
                    for user in uids2:
                        u1_temp = core_user    # 每次都要重新赋值
                        if core_user > user:
                            u1_temp, user = user, u1_temp
                        if self.u1_u2_sim.xs(u1_temp).xs(user)[0] < m:
                            break
                        num += 1
                    if len(uids2) == num:  # 如果满足相似度条件，则将该社区加入核心社区
                        self.user_cluster_list.loc[self.user_cluster_list.uid == core_user1, "is_core"] = 0
                        for user in uids2:
                            self.user_cluster_list.loc[self.user_cluster_list.uid == user, 'clusterId'] = clusterID
                        visited_cluster_temp.append(clusterid)
                        all_uids2.extend(uids2)
                for clusterid in visited_cluster_temp:
                    clusterlist.remove(clusterid)
                visited_cluster.extend(visited_cluster_temp)
                self.cal_community_l_feature(uids1, all_uids2, q, l, p_nums, uids1_resemblelist)

    def combinate_community(self, q, l, m):
        self.unsatisfied_clusterId = deepcopy(self.satisfied_clusterId[self.satisfied_clusterId.p_nums < l].reset_index(drop=True))  # 不满足特征l-多样性的社区
        self.unsatisfied_clusterId = self.unsatisfied_clusterId.sort_values(by=['p_nums'], ascending=False).reset_index(drop=True)  # 按照p特征的个进行降序排序
        self.satisfied_clusterId = deepcopy(self.satisfied_clusterId[self.satisfied_clusterId.p_nums >= l].reset_index(drop=True))  # 记录满足条件的社区
        # print("满足条件社区个数", len(self.satisfied_clusterId))
        # print("不满足条件社区个数", len(self.unsatisfied_clusterId))
        # print(len(self.user_cluster_list))
        while m > 0.1:  # 进行社区合并，使得社区内p特征的个数为L个，但是社区内不满足self.u1_u2_sim的社区相似度
            self.unsatisfied_clusterId_temp = []
            m = m - 0.1
            self.run_sim(q, l, m)  # 社区合并过程
            self.unsatisfied_clusterId_temp = pd.DataFrame(self.unsatisfied_clusterId_temp, columns=['clusterId', 'uids_nums','p_nums', 'uids', 'resemble_list'])  # 记录每次社区合并的中间结果
            self.satisfied_clusterId = pd.concat([self.satisfied_clusterId, self.unsatisfied_clusterId_temp[self.unsatisfied_clusterId_temp.p_nums >= l]], ignore_index=True)  # 记录合并社区后满足L多样性的社区
            self.unsatisfied_clusterId = deepcopy(self.unsatisfied_clusterId_temp[self.unsatisfied_clusterId_temp.p_nums < l])  # 将中间结果中不满足L多样性的社区提取出来，便于进行下一次社区合并
            self.unsatisfied_clusterId = self.unsatisfied_clusterId.sort_values(by=['p_nums'], ascending=False).reset_index(drop=True)  # 按照p_nums个数进行降序排列
            # print("满足条件社区个数", len(self.satisfied_clusterId))
            # print("不满足条件社区个数", len(self.unsatisfied_clusterId))
        # uids_nums1 = list(self.unsatisfied_clusterId.uids_nums.values)
        # uids_nums2 = list(self.satisfied_clusterId.uids_nums.values)
        # print("合并社区中用户的总数量", sum(uids_nums1) + sum(uids_nums2))

    def combinate_community_last(self, q, l):
        self.unsatisfied_clusterId = self.unsatisfied_clusterId.sort_values(by=['p_nums'], ascending=False).reset_index(drop=True)  # 按照p特征的个进行降序排序
        # print("满足条件社区个数", len(self.satisfied_clusterId))
        # print("不满足条件社区个数", len(self.unsatisfied_clusterId))
        while True:  # 进行社区合并，使得社区内p特征的个数为L个，但是社区内不满足self.u1_u2_sim的社区相似度
            self.unsatisfied_clusterId_temp = []
            flag = self.run(q, l)  # 社区合并过程
            if flag is False:
                self.unsatisfied_clusterId_temp = pd.DataFrame(self.unsatisfied_clusterId_temp, columns=['clusterId', 'uids_nums','p_nums', 'uids', 'resemble_list'])  # 记录每次社区合并的中间结果
                self.satisfied_clusterId = pd.concat([self.satisfied_clusterId, self.unsatisfied_clusterId_temp[self.unsatisfied_clusterId_temp.p_nums >= l]], ignore_index=True)  # 记录合并社区后满足L多样性的社区
                self.unsatisfied_clusterId = deepcopy(self.unsatisfied_clusterId_temp[self.unsatisfied_clusterId_temp.p_nums < l])  # 将中间结果中不满足L多样性的社区提取出来，便于进行下一次社区合并
                self.unsatisfied_clusterId = self.unsatisfied_clusterId.sort_values(by=['p_nums'], ascending=False).reset_index(drop=True)  # 按照p_nums个数进行降序排列
            else:
                break
        # print("满足条件社区个数", len(self.satisfied_clusterId))
        # print("不满足条件社区个数", len(self.unsatisfied_clusterId))
        # uids_nums1 = list(self.unsatisfied_clusterId.uids_nums.values)
        # uids_nums2 = list(self.satisfied_clusterId.uids_nums.values)
        # print("合并社区中用户的总数量", sum(uids_nums1) + sum(uids_nums2))

    def add_dummyuser(self, uids, q):
        different_feature_user = []  # 每个用户的不同特征的用户
        for user in uids:
            add_users = self.u1_u2_weighted_sim_list[((self.u1_u2_weighted_sim_list.u1 == user) | (self.u1_u2_weighted_sim_list.u2 == user))]
            add_users = add_users.sort_values(by=['similarity'], ascending=False).reset_index(drop=True)
            add_users = add_users[add_users.similarity <= q]
            add_uids = set.union(set(add_users['u1'].values), set(add_users['u2'].values))
            add_uids.remove(user)
            different_feature_user.append(add_uids)
        feature_users = different_feature_user[0]
        for i in range(len(different_feature_user)):
            if i > 0:
                feature_users = feature_users.intersection(different_feature_user[i])
        feature_users = list(feature_users)  # 满足条件的虚假用户记录
        if len(feature_users) == 0:
            return -1
        return feature_users[0]

    def community_divide_core(self, m, l, q, a, b, method, k, times):   # 用户隐私需求社区中至少包含n个不同的特征,相似度参数
        self.clusterId = 0  # 初始聚类号
        user_cluster_list = []  # 用于记录每个用户属于的聚类，格式为[user,clusterId，is_core]
        for user in self.users:
             tmp = [user, -1, 0]
             user_cluster_list.append(tmp)
        self.user_cluster_list = pd.DataFrame(user_cluster_list, columns=["uid", "clusterId", "is_core"])
        visitedusers = []              # 记录已经划分社区的用户
        self.satisfied_clusterId = []  # 记录满足所有社区中满足以及不满足特征 p 有多少个，格式为[clusterId,user_nums,p_nums],[社区id,用户数, 特征个数]
        print("第一步: 获取需要使用的特征相似度值")
        if method == "comloc":  # 共同访问位置
            self.u1_u2_weighted_sim_list = pd.read_csv(self.pairs_path + self.city + "_freqloc.similarity", names=["u1", "u2", "similarity"], header=None)
            self.u1_u2_sim = pd.read_csv(self.pairs_path + self.city + "_comloc.similarity", names=["u1", "u2", "similarity"], header=None)
            self.u1_u2_weighted_sim = deepcopy(self.u1_u2_weighted_sim_list).set_index(["u1", "u2"])  # 将相似度的u1和u2作为索引，取值更方便,并且不改变原有值数据框
        elif method == "freqloc":  # 频繁访问位置
            self.u1_u2_weighted_sim_list = pd.read_csv(self.pairs_path + self.city + "_comloc.similarity", names=["u1", "u2", "similarity"], header=None)
            self.u1_u2_sim = pd.read_csv(self.pairs_path + self.city + "_freqloc.similarity", names=["u1", "u2", "similarity"], header=None)
            self.u1_u2_weighted_sim = deepcopy(self.u1_u2_weighted_sim_list).set_index(["u1", "u2"])  # 将相似度的u1和u2作为索引，取值更方便,并且不改变原有值数据框
        self.checkins.loc[:, 'clusterid'] = -1  # 初始化社区id, -1表示该用户记录还没有划分到具体社区
        print("第二步: 开始计算每个用户属于的聚类")
        # 2.计算每个用户属于的聚类,m为内部参数用来判定用户的邻居 ,用户隐私需求参数 n
        users = deepcopy(self.users)
        while len(users) > 0:
            user = np.random.choice(users)
            user_k_users = self.u1_u2_sim[((self.u1_u2_sim.u1 == user) | (self.u1_u2_sim.u2 == user))]   # 获得与user相关的所有用户对
            user_k_users = user_k_users.sort_values(by=["similarity"], ascending=False).reset_index(drop=True)  # 按照亲密度进行降序排序
            user_sim_users = user_k_users[user_k_users.similarity >= m]  # 选取亲密度>=m的用户对记录
            uids = set.union(set(user_sim_users['u1'].values), set(user_sim_users['u2'].values)) # 当前社区内的所有用户
            if len(uids) != 0:
                uids.remove(user)
            uids = list(uids - set(visitedusers))  # 过滤掉已经划分社区的用户，set转成list顺序可能发生变化
            uids.insert(0, user)    # 将核心用户放在第一个
            self.user_cluster_list.loc[self.user_cluster_list.uid == user, 'is_core'] = 1
            self.cal_l_feature(uids, q, l)         # 计算社区中需要发布的特征有多少个，并用 self.satisfied_clusterId进行记录
            visitedusers.extend(uids)
            users = list(set(users) - set(uids))
        self.satisfied_clusterId = pd.DataFrame(self.satisfied_clusterId, columns=['clusterId', 'uids_nums', 'p_nums', 'uids', 'resemble_list'])
        # uids_nums = list(self.satisfied_clusterId.uids_nums.values)
        # print("验证是否访问所有用户以及所有社区用户个数之和", len(visitedusers), sum(uids_nums))
        # temp = self.user_cluster_list[self.user_cluster_list.is_core == 1]
        # print("核心用户的个数以及社区个数", len(temp), len(self.satisfied_clusterId))
        print("第三步: 将社区内合特征个数不满足l个的社区进行合并")
        self.u1_u2_sim = deepcopy(self.u1_u2_sim).set_index(["u1", "u2"])    # 重新指定索引，便于在社区合并的时候进行相似度的判断
        self.combinate_community(q, l, m)

        # 合并之后最后一个社区可能不满足特征L多样性，因此还需要添加虚假用户
        if len(self.unsatisfied_clusterId) > 1:   # 按照社区相似度划分社区后社区个数仍不止一个,还要进行合并
            self.combinate_community_last(q, l)
        checkins = pd.DataFrame()
        for row in self.satisfied_clusterId.itertuples(index=False, name=False):
            user_list = row[3]
            clusterid = row[0]
            for user in user_list:
                user_checkins = deepcopy(self.checkins[self.checkins.uid == user])
                user_checkins.loc[:, 'clusterid'] = clusterid
                checkins = pd.concat([checkins, user_checkins], ignore_index=True)
            checkins.reset_index(drop=True)
        if len(self.unsatisfied_clusterId) != 0:  # 需要添加虚假用户的社区
            temp_checkins = pd.DataFrame()
            clusterId_last = self.unsatisfied_clusterId.clusterId.values[0]
            p_nums = self.unsatisfied_clusterId.p_nums.values[0]
            uids = deepcopy(self.unsatisfied_clusterId.uids.values[0])
            print("最后一个社区不满足要求", p_nums)
            for user in uids:
                user_checkins = deepcopy(self.checkins[self.checkins.uid == user])
                user_checkins.loc[:, 'clusterid'] = clusterId_last
                temp_checkins = pd.concat([temp_checkins, user_checkins], ignore_index=True)
            temp_checkins.reset_index(drop=True)    # 最后一个社区的用户签到记录
            feature_users = []
            while len(feature_users) < l-p_nums:  # 一定要满足
                uid = self.add_dummyuser(uids, q)
                if uid == -1:
                    break
                uids.append(uid)
                feature_users.append(uid)
            dummy_users = []
            for i in range(l-p_nums):
                dummy_users.append(np.random.randint(5000000, 6000000))  # 随机产生lack_p_nums个用户
            uids = self.unsatisfied_clusterId.uids.values[0]
            uids.extend(dummy_users)
            if len(feature_users) == l-p_nums:
                print("有足够的虚假特征")
                i = 0
                for user in feature_users:
                    u_checkins = deepcopy(self.checkins[self.checkins.uid == user])
                    u_checkins.loc[:, 'clusterid'] = clusterId_last
                    u_checkins.loc[:, 'uid'] = dummy_users[i]
                    i += 1
                    temp_checkins = pd.concat([temp_checkins, u_checkins], ignore_index=True).reset_index(drop=True)
                checkins = pd.concat([checkins, temp_checkins], ignore_index=True).reset_index(drop=True)
                self.unsatisfied_clusterId.loc[:, 'p_nums'] = l
                self.satisfied_clusterId = pd.concat([self.satisfied_clusterId, self.unsatisfied_clusterId], ignore_index=True).reset_index(drop=True)
            else:
                print("没有足够的虚假特征")
        del self.unsatisfied_clusterId
        del self.unsatisfied_clusterId_temp
        self.checkins = self.checkins.ix[:, [0, 1, 2, 3, 4, 5]]
        gc.collect()
        print("第四步: 进行特征扰动并进行安全性和可靠性计算")
        core_uids = self.user_cluster_list[self.user_cluster_list.is_core == 1].uid.values
        if len(checkins) == 0:
            print("您的数据不能满足多样性！")
            return
        if len(checkins.clusterid.unique()) == 1:
            print("您的数据只能划分成一个社区，请修改参数")
            return
        print(len(core_uids), len(checkins.clusterid.unique()))  # 跟社区个数相同
        checkins.loc[:, 'is_core'] = 0
        for user in core_uids:
            checkins.loc[checkins.uid == user, 'is_core'] = 1
        self.clusterlist = [row[0] for row in self.satisfied_clusterId.itertuples(index=False, name=False)]
        self.is_resemble = [row[4] for row in self.satisfied_clusterId.itertuples(index=False, name=False)]
        if method == "comloc":  # 共同访问位置
            # 特征多样性l  method:社区划分后的文件名 m:扰动特征相似度阈值
            pc = disturb_comloc(l, method, m, self.path, self.lons_per_km, self.lats_per_km, self.city+"_"+str(m)+"_"+str(q)+"_"+str(times))
            ano_checkins = pc.comnunity_disturb(checkins, k)  # k为前k个频繁访问位置
            commutity_se = commutity_security("comloc", l, self.city, self.lats_per_km, self.lons_per_km)
            commutity_se.set_checkins(ano_checkins, self.checkins)
            commutity_se.cal_security(str(l))
        elif method == "freqloc":  # 频繁访问位置
            pc = disturb_freqloc(l, method, self.path, self.city+"_"+str(m)+"_"+str(q)+"_"+ str(times), self.M, self.N, m, self.clusterlist, self.is_resemble, self.satisfied_clusterId.ix[:, [0, 2]], self.latInterval, self.lngInterval, self.lons_per_km, self.lats_per_km)    # 参数为（l多样性、网格个数m,n、相似度m）
            ano_checkins = pc.comnunity_disturb(checkins, k)  # k为前k个频繁访问位置
            commutity_se = commutity_security("freqloc", l, self.city, self.lats_per_km, self.lons_per_km)
            commutity_se.set_checkins(ano_checkins, self.checkins)
            commutity_se.cal_security(str(l))


if __name__ == "__main__":
    start = time.time()
    argv = [0.6, 0.6, 0.6, 3]
    deta = math.exp(-1/3)
    t = math.exp(-0.8)
    l = argv[3]
    # for l in [3, 4, 5, 6, 7, 8, 9]:
    for l in [9]:
        for i in [83, 84, 85]:
        #for a in [0.1, 0.2, 0.3, 0.4, 0.7, 0.8]:
        # for a in [0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54]:
        # for a in [0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.601, 0.602, 0.603, 0.604, 0.605, 0.609, 0.61, 0.62, 0.7, 0.75, 0.8]:
            start1 = time.time()
            test = community_divide2()
            # 奥斯汀
            test.set_checkins("G:/pyfile/relation_protect/src/data/", "1", 30, 40, [30.387953, -97.843911, 30.249935, -97.635460])
            test.community_divide_core(math.exp(-1/2), l, math.exp(-1/2), 0.5, 0.5, "comloc", 3, i) #500
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
            print(str(time.time()-start1))
    end = time.time()
    print(str(end-start))