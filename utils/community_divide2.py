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


class community_divide2():

    def __init__(self):
        pass

    def set_checkins(self, path, city, n, m):
        data_processor = data_process()
        data_processor.set_basic_info(path, city)
        self.pairs_path = data_processor.pairs_path
        self.path = data_processor.path
        self.city = data_processor.city
        users = data_processor.checkins.uid.unique().tolist()
        users.sort(reverse=False)
        self.users = np.array(deepcopy(users))  # 将用户的id进行从小到大的排序
        del users
        gc.collect()
        self.m = m    # 经度网格
        self.n = n    # 纬度网格
        grid_divider = grid_divide(data_processor.checkins.values.tolist(), self.n, self.m, [30.387953, -97.843911, 30.249935, -97.635460])
        self.checkins = grid_divider.divide_area_by_NN().ix[:, [0, 1, 2, 3, 4, 5]]  # 带有网格id
        self.is_resemble = []  # 用来记录社区内相似用户对
        self.clusterlist = []  # 用来记录社区id
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
                # sim = self.u1_u2_weighted_sim.xs(u1).xs(u2_temp)[0]
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
                    # self.checkins.loc[self.checkins.uid == u, 'clusterid'] = self.clusterId
                self.satisfied_clusterId.append([self.clusterId, len(uid_temp), num-1, uid_temp])
                self.clusterlist.append(self.clusterId)  # 一一对应关系
                self.is_resemble.append(resemble_list)
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
            # if num == l:
            #     for u in uid_temp:
            #         self.checkins.loc[self.checkins.uid == u, 'clusterid'] = self.clusterId
            self.satisfied_clusterId.append([self.clusterId, len(uid_temp), num, uid_temp])
            self.clusterlist.append(self.clusterId)  # 一一对应关系
            self.is_resemble.append(resemble_list)
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
        # uid_temp.sort(reverse=False)
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
                    # self.is_resemble.append([self.clusterId, u1, u2_temp])
                    resemble_list.append([u1_temp, u2])
                    break
                temp_num += 1
            if len(uid_temp) == temp_num:
                num += 1
            uid_temp.append(u2)
            uids_temp.remove(uids_temp[0])
            # if num == l+1:  # 说明当前至少有不同的特征l个，则划分一个社区
            if num == 2*l:
                uids_temp.insert(0, u2)
                uid_temp.remove(u2)
                for u in uid_temp:
                    self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
                # 记录社区划分的中间结果
                self.unsatisfied_clusterId_temp.append([self.clusterId, len(uid_temp), num-1, uid_temp])
                self.clusterlist.append(self.clusterId)  # 一一对应关系
                self.is_resemble.append(resemble_list)
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
            self.unsatisfied_clusterId_temp.append([self.clusterId, len(uid_temp), num, uid_temp])
            self.clusterlist.append(self.clusterId)  # 一一对应关系
            self.is_resemble.append(resemble_list)
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
            # uids2 = list(self.user_cluster_list[self.user_cluster_list.clusterId == tail_clusterid].uid.values)
            # uids1 = list(self.user_cluster_list[self.user_cluster_list.clusterId == header_clusterId].uid.values)
            uids2_core = self.user_cluster_list[(self.user_cluster_list.clusterId == tail_clusterid) & (self.user_cluster_list.is_core == 1)].uid.values[0]
            self.user_cluster_list.loc[self.user_cluster_list.uid == uids2_core, 'is_core'] = 0  # 将需要合并的尾社区的核心用户取消标记
            index = self.clusterlist.index(header_clusterId)
            uids1_resemblelist = self.is_resemble[index]
            index2 = self.clusterlist.index(tail_clusterid)
            if index > index2:
                del self.is_resemble[index]
                del self.is_resemble[index2]
            else:
                del self.is_resemble[index2]
                del self.is_resemble[index]
            self.clusterlist.remove(header_clusterId)
            self.clusterlist.remove(tail_clusterid)

            p_nums = self.unsatisfied_clusterId.loc[i, 'p_nums']
            self.cal_community_l_feature(uids1, uids2, q, l, p_nums, uids1_resemblelist)
        return False

    def combinate_community(self, q, l):
        self.unsatisfied_clusterId = deepcopy(self.satisfied_clusterId[self.satisfied_clusterId.p_nums < l].reset_index(drop=True))  # 不满足特征l-多样性的社区
        self.unsatisfied_clusterId = self.unsatisfied_clusterId.sort_values(by=['p_nums'], ascending=False).reset_index(drop=True)  # 按照p特征的个进行降序排序
        self.satisfied_clusterId = deepcopy(self.satisfied_clusterId[self.satisfied_clusterId.p_nums >= l].reset_index(drop=True))  # 记录满足条件的社区

        print(len(self.user_cluster_list))
        # satistfied_clusterId = pd.DataFrame()
        while True:  # 进行社区合并，使得社区内p特征的个数为L个，但是社区内不满足self.u1_u2_sim的社区相似度
            self.unsatisfied_clusterId_temp = []
            flag = self.run(q, l)  # 社区合并过程
            if flag is False:
                self.unsatisfied_clusterId_temp = pd.DataFrame(self.unsatisfied_clusterId_temp, columns=['clusterId', 'uids_nums','p_nums', 'uids'])  # 记录每次社区合并的中间结果
                self.satisfied_clusterId = pd.concat([self.satisfied_clusterId, self.unsatisfied_clusterId_temp[self.unsatisfied_clusterId_temp.p_nums >= l]], ignore_index=True)  # 记录合并社区后满足L多样性的社区
                self.unsatisfied_clusterId = deepcopy(self.unsatisfied_clusterId_temp[self.unsatisfied_clusterId_temp.p_nums < l])  # 将中间结果中不满足L多样性的社区提取出来，便于进行下一次社区合并
                self.unsatisfied_clusterId = self.unsatisfied_clusterId.sort_values(by=['p_nums'], ascending=False).reset_index(drop=True)  # 按照p_nums个数进行降序排列
            else:
                break
        uids_nums1 = list(self.unsatisfied_clusterId.uids_nums.values)
        uids_nums2 = list(self.satisfied_clusterId.uids_nums.values)
        print("合并社区中用户的总数量", sum(uids_nums1) + sum(uids_nums2))

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
        return feature_users[0]

    def community_divide_core(self, m, l, q, a, b, method, k):   # 用户隐私需求社区中至少包含n个不同的特征,相似度参数
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
        for user in self.users:
            if user not in visitedusers:  # 如果用户还没有被访问过，也就是没有划分社区的用户
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
        # self.user_cluster_list.to_csv("G:/pyfile/relation_protect/src/data/city_data/" + self.city + "_" + method + "_" + str(l) + "_user_cluster_list.csv", sep='\t', header=False, index=False)
        self.satisfied_clusterId = pd.DataFrame(self.satisfied_clusterId, columns=['clusterId', 'uids_nums', 'p_nums', 'uids'])
        uids_nums = list(self.satisfied_clusterId.uids_nums.values)
        print("验证是否访问所有用户以及所有社区用户个数之和", len(visitedusers), sum(uids_nums))
        # self.satisfied_clusterId.to_csv(self.pairs_path + self.city + "_" + method + "_" + str(l) + "_satisfied_clusterId.csv", index=False, header=False)
        temp = self.user_cluster_list[self.user_cluster_list.is_core == 1]
        print("核心用户的个数以及社区个数", len(temp), len(self.satisfied_clusterId))

        # checkins_undivide = deepcopy(self.checkins[self.checkins.clusterid == -1])  # 还没有进行划分社区的用户记录
        # checkins_divided = self.checkins[~self.checkins.clusterid.isin([-1])]  # 先将没有划分社区的用户记录删除，并将该纪录存入文件
        print("第三步: 将社区内合特征个数不满足l个的社区进行合并")
        self.combinate_community(q, l)
        # for row in self.user_cluster_list.itertuples(index=False, name=False):  # 对社区合并之前还没有划分社区的用户进行社区划分
            # checkins_undivide.loc[checkins_undivide.uid == row[0], 'clusterid'] = row[1]
        # 合并之后最后一个社区可能不满足特征L多样性，因此还需要添加虚假用户
        checkins = pd.DataFrame()
        for row in self.satisfied_clusterId.itertuples(index=False,name=False):
            user_list = row[3]
            clusterid = row[0]
            for user in user_list:
                user_checkins = deepcopy(self.checkins[self.checkins.uid == user])
                user_checkins.loc[:, 'clusterid'] = clusterid
                checkins = pd.concat([checkins, user_checkins], ignore_index=True)
            checkins.reset_index(drop=True)
        if len(self.unsatisfied_clusterId) != 0:
            temp_checkins = pd.DataFrame()
            clusterId_last = self.unsatisfied_clusterId.clusterId.values[0]
            p_nums = self.unsatisfied_clusterId.p_nums.values[0]
            uids = deepcopy(self.unsatisfied_clusterId.uids.values[0])
            print(uids)
            print("最后一个社区不满足要求", p_nums)
            for user in uids:
                user_checkins = deepcopy(self.checkins[self.checkins.uid == user])
                user_checkins.loc[:, 'clusterid'] = clusterId_last
                temp_checkins = pd.concat([temp_checkins, user_checkins], ignore_index=True)
            temp_checkins.reset_index(drop=True)    # 最后一个社区的用户签到记录
            # uids = list(self.user_cluster_list[self.user_cluster_list.clusterId == clusterId_last].uid.unique())
            # 添加虚假用户，修改self.user_cluster_list和 satistfied_clusterId
            feature_users = []
            while len(feature_users) < l-p_nums:  # 一定要满足
            # for i in range(l-p_nums):
                uid = self.add_dummyuser(uids, q)
                uids.append(uid)
                feature_users.append(uid)
            # lack_p_nums = l-p_nums

            dummy_users = []
            for i in range(l-p_nums):
                dummy_users.append(np.random.randint(5000000, 6000000))  # 随机产生lack_p_nums个用户
            uids = self.unsatisfied_clusterId.uids.values[0]
            uids.extend(dummy_users)
            print(self.unsatisfied_clusterId.uids.values[0])
            if len(feature_users) != 0:
                i = 0
                for user in feature_users:
                    u_checkins = deepcopy(self.checkins[self.checkins.uid == user])
                    u_checkins.loc[:, 'clusterid'] = clusterId_last
                    u_checkins.loc[:, 'uid'] = dummy_users[i]
                    i += 1
                    temp_checkins = pd.concat([temp_checkins, u_checkins], ignore_index=True).reset_index(drop=True)
                checkins = pd.concat([checkins, temp_checkins], ignore_index=True).reset_index(drop=True)
                self.unsatisfied_clusterId.loc[:, 'p_nums'] = l
                # self.unsatisfied_clusterId.loc[:, 'uids'] = uids
            # else:
            #     checkins_undivide = checkins_undivide[~checkins_undivide.clusterid.isin([clusterId_last])]
            # checkins_undivide = checkins_undivide.reset_index(drop=True)
        self.satisfied_clusterId = pd.concat([self.satisfied_clusterId, self.unsatisfied_clusterId], ignore_index=True).reset_index(drop=True)
        del self.unsatisfied_clusterId
        del self.unsatisfied_clusterId_temp
        del self.checkins

        gc.collect()
        print("社区合并之后社区个数：", len(self.clusterlist))
        print("第四步: 进行特征扰动")
        print("合并之后用户数量", len(checkins.uid.unique()))
        # checkins_divided = checkins_divided.sort_values(by=['uid'], ascending=True).reset_index(drop=True)
        # checkins_undivide = checkins_undivide.sort_values(by=['uid'], ascending=True).reset_index(drop=True)
        # uids1 = checkins_divided.uid.unique()
        # uids2 = checkins_undivide.uid.unique()
        # all_user = list(set(uids1).union(set(uids2)))
        # print("验证用户并集与交集个数", len(all_user), len(set(uids1).intersection(set(uids2))))
        # checkins_divided = pd.concat([checkins_divided, checkins_undivide], ignore_index=True).reset_index(drop=True)
        # 重新计算一次相似度
        core_uids = self.user_cluster_list[self.user_cluster_list.is_core == 1].uid.values
        # print(len(core_uids), len(checkins_divided.clusterid.unique()))  # 跟社区个数相同
        # checkins_divided.loc[:, 'is_core'] = 0
        # for user in core_uids:
        #     checkins_divided.loc[checkins_divided.uid == user, 'is_core'] = 1
        print(len(core_uids), len(checkins.clusterid.unique()))  # 跟社区个数相同
        checkins.loc[:, 'is_core'] = 0
        for user in core_uids:
            checkins.loc[checkins.uid == user, 'is_core'] = 1
        ano_checkins = pd.DataFrame()
        if method == "comloc":  # 共同访问位置
            pc = disturb_comloc(l, method, m)
            ano_checkins = pc.comnunity_disturb(checkins, k)  # k为前k个频繁访问位置
            # cal_similarity1 = cal_similarity(method)
            # ano_checkin = ano_checkins.ix[:, [0, 1, 2, 3, 4]]
            # cal_similarity1.set_checkins(ano_checkin.values.tolist(), "1", l, 30, 40, [30.387953, -97.843911, 30.249935, -97.635460])
            # self.u1_u2_sim = cal_similarity1.cal_comloc_user_pairs()
            # self.u1_u2_weighted_sim_list = cal_similarity1.cal_freqloc_user_pairs(k)
        elif method == "freqloc":  # 频繁访问位置
            pc = disturb_freqloc(l, method, self.m, self.n, m, self.clusterlist, self.is_resemble, l, self.satisfied_clusterId.ix[:, [0, 2]])    # 参数为（l多样性、网格个数m,n、相似度m）
            ano_checkins = pc.comnunity_disturb(checkins, k)  # k为前k个频繁访问位置
            # cal_similarity1 = cal_similarity(method)
            # ano_checkin = ano_checkins.ix[:, [0, 1, 2, 3, 4]]
            # cal_similarity1.set_checkins(ano_checkin.values.tolist(), "1", l, 30, 40, [30.387953, -97.843911, 30.249935, -97.635460])
            # self.u1_u2_sim = cal_similarity1.cal_comloc_user_pairs()
            # self.u1_u2_weighted_sim_list = cal_similarity1.cal_freqloc_user_pairs(k)

        # self.u1_u2_weighted_sim = deepcopy(self.u1_u2_weighted_sim_list).set_index(["u1", "u2"])  # 将相似度的u1和u2作为索引，取值更方便,并且不改变原有值数据框
        # self.u1_u2_sim = deepcopy(self.u1_u2_sim).reset_index(["u1", "u2"])
        # l_feature = cal_l_fetaures("G:/pyfile/relation_protect/src/data/result_data/", "1")
        # unsatisfied_clusters = l_feature.test(k, m, q, ano_checkins, self.u1_u2_weighted_sim, self.u1_u2_sim)   # 返回clusterid以及p特征个数
        # clusterids = list(unsatisfied_clusters.clusterid.unique())
        # unsatisfied_ano_checkins = ano_checkins[ano_checkins.clusterid.isin(clusterids)]
        # self.user_cluster_list = unsatisfied_ano_checkins.ix[:, [0, 5]]
        # self.user_cluster_list.drop_duplicates(subset=None, keep='first', inplace=True)  # 去掉重复行
        # self.satisfied_clusterId =
        print("第五步: 安全性和可靠性计算")


if __name__ == "__main__":
    start = time.time()
    # for i in [3]:
    for i in [3, 4, 5, 6, 7, 8, 9, 10]:
        start1 = time.time()
        test = community_divide2()
        test.set_checkins("G:/pyfile/relation_protect/src/data/city_data/", "1", 30, 40)
        # test.community_divide_core(math.exp(-1/3), i, math.exp(-1/2), 0.5, 0.5, "comloc", 3)
        test.community_divide_core(math.exp(-1), i, math.exp(-1/3), 0.5, 0.5, "freqloc", 3)
        print(str(time.time()-start1))
    end = time.time()
    print(str(end-start))