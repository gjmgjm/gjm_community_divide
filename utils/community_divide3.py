#!/usr/bin/env python
# encoding: utf-8

import time
from data_process import data_process
from copy import deepcopy
import pandas as pd
import numpy as np
import gc


class community_divide3():

    def __init__(self):
        pass

    def set_checkins(self, path, city):
        data_processor = data_process()
        data_processor.set_basic_info(path, city)
        # data_processor.user_pairs()
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

    def cal_l_feature(self, uids, q, l):  # 方法与社区合并类似
        uids_temp = deepcopy(uids)
        uid_temp = [uids_temp[0]]
        uids_temp.remove(uids_temp[0])
        num = 1
        while len(uids_temp) > 0:
            u2 = uids_temp[0]
            temp_num = 0
            for u1 in uid_temp:
                if self.u1_u2_weighted_sim.xs(u1).xs(u2)[0] > q:  # 不满足l多样性
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
                    self.checkins.loc[self.checkins.uid == u, 'clusterid'] = self.clusterId
                self.satisfied_clusterId.append([self.clusterId, len(uid_temp), num-1])
                self.clusterId += 1
                if len(uids_temp) > 0:
                    uid_temp = [uids_temp[0]]
                    uids_temp.remove(uids_temp[0])
                    num = 1
        if len(uid_temp) != 0:  # 最后可能还是有没有l个特征的社区
            for u in uid_temp:
                self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
            if num == l:
                for u in uid_temp:
                    self.checkins.loc[self.checkins.uid == u, 'clusterid'] = self.clusterId
            self.satisfied_clusterId.append([self.clusterId, len(uid_temp), num])
            self.clusterId += 1

    def cal_community_l_feature(self, uids1, uids2, q, l, p_nums):
        """
        :param uids1: 首社区中用户
        :param uids2: 尾社区中用户
        :param q:    社区l多样性的判断条件，即相似度值
        :param l:    社区特征多样性个数l
        :param p_nums:  首社区中p特征的个数
        :return:
        依次将尾社区中的元素添加到首社区中，并判断p特征的个数，当p特征的个数等于l时，进行一次社区分裂
        """
        uids_temp = deepcopy(uids2)
        uid_temp = deepcopy(uids1)
        uid_temp.sort(reverse=False)
        num = p_nums
        while len(uids_temp) > 0:
            u2 = uids_temp[0]
            temp_num = 0
            for u1 in uid_temp:
                u2_temp = u2
                if u1 > u2:
                    u1, u2_temp = u2_temp, u1
                if self.u1_u2_weighted_sim.xs(u1).xs(u2_temp)[0] > q:  # 不满足l多样性
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
                self.unsatisfied_clusterId_temp.append([self.clusterId, len(uid_temp), num-1])
                self.clusterId += 1
                if len(uids_temp) > 0:    # 进行社区分裂之后，尾社区还有元素，进行下一次合并
                    uid_temp = [uids_temp[0]]
                    uids_temp.remove(uids_temp[0])
                    num = 1
        if len(uid_temp) != 0:  # 记录最后一个没有进行社区分裂的社区
            for u in uid_temp:
                self.user_cluster_list.loc[self.user_cluster_list.uid == u, 'clusterId'] = self.clusterId
            self.unsatisfied_clusterId_temp.append([self.clusterId, len(uid_temp), num])
            self.clusterId += 1

    def run(self, q, l):
        """
        :param q:  社区l多样性的判断条件，即相似度值
        :param l:  社区特征多样性个数l
        :return:
        """
        community_divide_count = self.unsatisfied_clusterId[self.unsatisfied_clusterId.p_nums != l].reset_index(drop=True)
        community_divide_count_LACK = self.unsatisfied_clusterId[self.unsatisfied_clusterId.p_nums < l].reset_index(drop=True)
        if (len(community_divide_count) == 1 and len(community_divide_count_LACK) == 1) or len(community_divide_count) == 0:
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
            uids2 = list(self.user_cluster_list[self.user_cluster_list.clusterId == tail_clusterid].uid.values)
            uids1 = list(self.user_cluster_list[self.user_cluster_list.clusterId == header_clusterId].uid.values)
            p_nums = self.unsatisfied_clusterId.loc[i, 'p_nums']
            self.cal_community_l_feature(uids1, uids2, q, l, p_nums)
        return False

    def community_divide_core(self, m, l, q, a, b, method, k):   # 用户隐私需求社区中至少包含n个不同的特征,相似度参数
        from cal_similarity import cal_similarity
        self.clusterId = 0  # 初始聚类号
        user_cluster_list = []  # 用于记录每个用户属于的聚类，格式为[user,clusterId]
        for user in self.users:
             tmp = [user, -1]
             user_cluster_list.append(tmp)
        self.user_cluster_list = pd.DataFrame(user_cluster_list, columns=["uid", "clusterId"])
        visitedusers = []              # 记录已经划分社区的用户
        self.satisfied_clusterId = []  # 记录满足所有社区中满足以及不满足特征 p 有多少个，格式为[clusterId,user_nums,p_nums],[社区id,用户数, 特征个数]

        print("第一步: 获取需要使用的特征相似度值并计算加权相似度")
        # if method == "comloc":  # 共同访问位置
        #     cal_similarity = cal_similarity()
        #     cal_similarity.set_checkins(self.checkins.values.tolist(), "1", 30, 40, [30.387953, -97.843911, 30.249935, -97.635460])
        #     self.u1_u2_weighted_sim = cal_similarity.cal_weighted_sim(k, a, b)
        #     # self.u1_u2_weighted_sim.to_csv(self.pairs_path + self.city + "_" + method + "_weighted.similarity", index=False, header=False)
        #     self.u1_u2_sim = cal_similarity.cal_comloc_user_pairs()
        #     # self.u1_u2_sim_comloc = pd.read_csv(self.pairs_path + self.city + "_comloc.similarity", names=["u1", "u2", "similarity"], header=None)
        #     self.u1_u2_weighted_sim = deepcopy(self.u1_u2_weighted_sim).set_index(["u1", "u2"])  # 将相似度的u1和u2作为索引，取值更方便,并且不改变原有值数据框
        # elif method == "freqloc":  # 频繁访问位置
        #     pass
        # else:                      # 位置访问频率分布
        #     pass

        # self.u1_u2_weighted_sim = pd.read_csv(self.pairs_path + self.city + "_" + method + "_weighted.similarity", header=None, names=["u1", "u2", "similarity"])
        self.u1_u2_weighted_sim = pd.read_csv(self.pairs_path + self.city + "_freqloc.similarity", names=["u1", "u2", "similarity"], header=None)
        self.u1_u2_sim = pd.read_csv(self.pairs_path + self.city + "_comloc.similarity", names=["u1", "u2", "similarity"], header=None)
        self.u1_u2_weighted_sim = deepcopy(self.u1_u2_weighted_sim).set_index(["u1", "u2"])  # 将相似度的u1和u2作为索引，取值更方便,并且不改变原有值数据框

        self.checkins.loc[:, 'clusterid'] = -1  # 初始化社区id, -1表示该用户记录还没有划分到具体社区
        print("第二步: 开始计算每个用户属于的聚类")
        # 2.计算每个用户属于的聚类,m为内部参数用来判定用户的邻居 ,用户隐私需求参数 n
        for user in self.users:
            if user not in visitedusers:  # 如果用户还没有被访问过，也就是没有划分社区的用户
                user_k_users = self.u1_u2_sim[((self.u1_u2_sim.u1 == user) | (self.u1_u2_sim.u2 == user))]   # 获得与user相关的所有用户对
                user_k_users = user_k_users.sort_values(by=["similarity"], ascending=False).reset_index(drop=True)  # 按照亲密度进行降序排序
                user_sim_users = user_k_users[user_k_users.similarity >= m]  # 选取亲密度>=m的用户对记录
                uids = set.union(set(user_sim_users['u1'].values), set(user_sim_users['u2'].values))  # 当前社区内的所有用户
                if len(uids) == 0:
                    uids.add(user)
                uids = list(uids - set(visitedusers))  # 过滤掉已经划分社区的用户
                uids.sort(reverse=False)
                self.cal_l_feature(uids, q, l)         # 计算社区中需要发布的特征有多少个，并用
                visitedusers.extend(uids)
        print(len(visitedusers))
        self.user_cluster_list.to_csv("G:/pyfile/relation_protect/src/data/city_data/" + self.city + "_" + method + "_" + str(l) + "_user_cluster_list.csv", sep='\t', header=False, index=False)
        self.satisfied_clusterId = pd.DataFrame(self.satisfied_clusterId, columns=['clusterId', 'uids_nums', 'p_nums'])
        uids_nums = list(self.satisfied_clusterId.uids_nums.values)
        print(sum(uids_nums))
        # self.satisfied_clusterId.to_csv(self.pairs_path + self.city + "_" + method + "_" + str(l) + "_satisfied_clusterId.csv", index=False, header=False)

        checkins = deepcopy(self.checkins[self.checkins.clusterid == -1])   # 还没有进行划分社区的用户记录
        self.checkins = self.checkins[~self.checkins.clusterid.isin([-1])]  # 先将没有划分社区的用户记录删除，并将该纪录存入文件
        # self.checkins.to_csv("G:/pyfile/relation_protect/src/data/result_data/" + method + "/" + self.city + "_" + method + "_" + str(l) + "_user_simple_community.checkins", index=False, sep='\t', header=False)

        print("第三步: 将社区内合特征个数不满足l个的社区进行合并")
        self.unsatisfied_clusterId = deepcopy(self.satisfied_clusterId[self.satisfied_clusterId.p_nums != l].reset_index(drop=True))  # 不满足特征l-多样性的社区
        self.unsatisfied_clusterId = self.unsatisfied_clusterId.sort_values(by=['p_nums'], ascending=False).reset_index(drop=True)    # 按照p特征的个进行降序排序
        clusterIds = self.unsatisfied_clusterId.clusterId.values    # 提取不满足特征l多样性社区
        # 可以简化成isin, 以后再来改
        user_cluster_list = pd.DataFrame()
        for clusterid in clusterIds:    # 将社区划分时社区内不满足l特征的用户记录下来，self.user_cluster_list
            user_cluster_list = pd.concat([user_cluster_list, self.user_cluster_list[self.user_cluster_list.clusterId == clusterid]], ignore_index=True)
        self.user_cluster_list = user_cluster_list.reset_index(drop=True)
        del clusterIds
        gc.collect()
        print(len(self.user_cluster_list))
        satistfied_clusterId = pd.DataFrame()
        while True:                # 进行社区合并，使得社区内p特征的个数为L个，但是社区内不满足self.u1_u2_sim的社区相似度
            self.unsatisfied_clusterId_temp = []
            flag = self.run(q, l)  # 社区合并过程
            if flag is False:
                self.unsatisfied_clusterId_temp = pd.DataFrame(self.unsatisfied_clusterId_temp, columns=['clusterId', 'uids_nums', 'p_nums'])  # 记录每次社区合并的中间结果
                satistfied_clusterId = pd.concat([satistfied_clusterId, self.unsatisfied_clusterId_temp[self.unsatisfied_clusterId_temp.p_nums == l]], ignore_index=True)  # 记录合并社区后满足L多样性的社区
                self.unsatisfied_clusterId = deepcopy(self.unsatisfied_clusterId_temp[self.unsatisfied_clusterId_temp.p_nums != l])       # 将中间结果中不满足L多样性的社区提取出来，便于进行下一次社区合并
                self.unsatisfied_clusterId = self.unsatisfied_clusterId.sort_values(by=['p_nums'], ascending=False).reset_index(drop=True)  # 按照p_nums个数进行降序排列
            else:
                break
        uids_nums1 = list(self.unsatisfied_clusterId.uids_nums.values)
        uids_nums2 = list(satistfied_clusterId.uids_nums.values)
        print(sum(uids_nums1) + sum(uids_nums2))

        for row in self.user_cluster_list.itertuples(index=False, name=False):  # 对社区合并之前还没有划分社区的用户进行社区划分
            checkins.loc[checkins.uid == row[0], 'clusterid'] = row[1]

        # 合并之后最后一个社区可能不满足特征L多样性，因此还需要添加虚假用户
        if len(self.unsatisfied_clusterId) != 0:
            clusterId_last = self.unsatisfied_clusterId.clusterId.values[0]
            checkins = checkins[~checkins.clusterid.isin([clusterId_last])]
        del self.unsatisfied_clusterId
        del self.unsatisfied_clusterId_temp
        gc.collect()
        print("第四步: 社区内经纬度精度下降")
        self.checkins = self.checkins.sort_values(by=['uid'], ascending=True).reset_index(drop=True)
        checkins = checkins.sort_values(by=['uid'], ascending=True).reset_index(drop=True)
        uids1 = self.checkins.uid.unique()
        uids2 = checkins.uid.unique()
        all_user = list(set(uids1).union(set(uids2)))
        print(len(all_user))
        # 进行社区内经纬度的经度下降
        ano_checkins = []
        community_checkins = self.checkins.groupby(['clusterid'])
        for group in community_checkins:
            # 进行经纬度的经度下降操作
            a = np.random.randint(3, 7)
            for row in group[1].itertuples(index=False, name=False):
                lat = round(row[2], a)
                lng = round(row[3], a)
                ano_checkin = [row[0], row[1], lat, lng, row[4], row[5]]
                ano_checkins.append(ano_checkin)
            # self.save_ano_checkins(ano_checkins, method, l)
        community_checkins = checkins.groupby(['clusterid'])
        for group in community_checkins:
            # 进行经纬度的经度下降操作
            a = np.random.randint(3, 7)
            # ano_checkins = []
            for row in group[1].itertuples(index=False, name=False):
                lat = round(row[2], a)
                lng = round(row[3], a)
                ano_checkin = [row[0], row[1], lat, lng, row[4], row[5]]
                ano_checkins.append(ano_checkin)
            # self.save_ano_checkins(ano_checkins, method, l)
        # 读取文件数据
        ano_checkins = pd.DataFrame(ano_checkins, columns=["uid", "time", "latitude", "longitude", "locid", "clusterid"])
        # all_checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/" + method + "/disturb_accuracy_" + str(l) + "_" + method+".csv", sep='\t', names=["uid", "time", "latitude", "longitude", "locid", "clusterid"], header=None)
        group_checkins = ano_checkins.groupby(['latitude', 'longitude'])
        uids_l = len(ano_checkins.uid.unique())
        print("用户数：", uids_l)
        locid = 0
        for group in group_checkins:
            checkin = deepcopy(group[1])
            checkin.loc[:, 'locid'] = locid
            locid += 1
            checkin.to_csv("G:/pyfile/relation_protect/src/data/result_data/" + method + "/1_disturb_accuracy_" + str(l) + "_" + method
                            +".csv", header=None, index=None, sep='\t', mode='a')

    def save_ano_checkins(self, ano_checkin, method, k):
        ano_checkins = pd.DataFrame(ano_checkin)
        ano_checkins.to_csv("G:/pyfile/relation_protect/src/data/result_data/" + method + "/disturb_accuracy_" + str(k) + "_" + method
                            +".csv", header=None, index=None, sep='\t', mode='a')


if __name__ == "__main__":
    start = time.time()
    # for i in [0.35, 0.3, 0.2, 0.19, 0.15, 0.13, 0.12, 0.1]:
    for i in [3, 4, 5, 6, 7, 8, 9, 10]:
        start1 = time.time()
        test = community_divide3()
        test.set_checkins("G:/pyfile/relation_protect/src/data/city_data/", "1")
        test.community_divide_core(0.54, i, 1/3, 0.5, 0.5, "comloc", 4)
        print(str(time.time()-start1))
    end = time.time()
    print(str(end-start))