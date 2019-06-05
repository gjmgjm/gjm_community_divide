#!/usr/bin/env python
# encoding: utf-8

from copy import deepcopy
import time
import pandas as pd
import numpy as np
import math


class cal_l_fetaures:

    def __init__(self, pair_path, city):
        self.pairs_path = pair_path
        self.city = city
        pass

    def cal_l_feature(self, uids, q, clusterid):  # 方法与社区合并类似
        uids_temp = deepcopy(uids)
        uid_temp = [uids_temp[0]]
        uids_temp.remove(uids_temp[0])
        num = 1
        while len(uids_temp) > 0:
            u2 = uids_temp[0]
            temp_num = 0
            for u1 in uid_temp:
                u2_temp = u2
                if u1 > u2:
                    u1, u2_temp = u2_temp, u1
                sim = self.u1_u2_weighted_sim.xs(u1).xs(u2_temp)[0]
                if self.u1_u2_weighted_sim.xs(u1).xs(u2_temp)[0] > q:  # 不满足l多样性
                    break
                temp_num += 1
            if len(uid_temp) == temp_num:
                num += 1
            uid_temp.append(u2)
            uids_temp.remove(uids_temp[0])
        return [clusterid, num]

    def cal_community_similarity(self, uids, m, clusterid):
        uids_temp = deepcopy(uids)
        u1 = uids_temp[0]     # 核心用户
        uids_temp.remove(u1)
        temp_num = 0
        for u2 in uids_temp:
            u1_temp = u1
            if u1 > u2:
                u1_temp, u2 = u2, u1_temp
            if self.u1_u2_sim.xs(u1_temp).xs(u2)[0] < m:   # 不满足特征相似性
                break
            temp_num += 1
        if len(uids_temp) == temp_num:
            return [clusterid, 1]
        return [clusterid, 0]

    def test(self, k, q, m, checkinss=None, u1_u2_weighted_sim=None, u1_u2_sim=None ):
        checkins_obf = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/SNAP_NY_1.csv", delimiter="\t", index_col=None)
        # checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/comloc/1_disturb_accuracy_" + str(k) + "_comloc.csv",
        #                        sep='\t', names=["uid", "time", "latitude", "longitude", "locid", "clusterid"],
        #                        header=None)
        checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/comloc/SNAP_NY_1_1_" + str(k) + "_comloc.csv",
            sep='\t', names=["uid", "time", "latitude", "longitude","locid", "grid_id", "clusterid", "is_core", "grid_id_before","locid_befroe"],
            header=None)
        # checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/freqloc/11_" + str(k) + "_freqloc.csv",
        #                        sep='\t', names=["uid", "time", "latitude", "longitude", "locid", "clusterid"],
        #                        header=None)
        uids_obf = set(checkins_obf.uid.unique())
        uids = set(checkins.uid.unique())
        cha = uids_obf - uids
        print(k, cha)
        print()
        u1_u2_weighted_sim = pd.read_csv(self.pairs_path+"comloc/" + self.city + "_freqloc_1_"+str(k)+"_after.similarity", names=["u1", "u2", "similarity"], header=None)
        u1_u2_sim = pd.read_csv(self.pairs_path+"comloc/" + self.city + "_comloc_1_"+str(k)+"_after.similarity", names=["u1", "u2", "similarity"], header=None)
        # u1_u2_sim = pd.read_csv(self.pairs_path + "freqloc/" + self.city + "_freqloc_4_" + str(k) + "_after.similarity", names=["u1", "u2", "similarity"], header=None)
        # u1_u2_weighted_sim = pd.read_csv(self.pairs_path + "freqloc/" + self.city + "_comloc_4_" + str(k) + "_after.similarity", names=["u1", "u2", "similarity"], header=None)

        self.u1_u2_weighted_sim = deepcopy(u1_u2_weighted_sim).set_index(["u1", "u2"])  # 将相似度的u1和u2作为索引，取值更方便,并且不改变原有值数据框
        self.u1_u2_sim = deepcopy(u1_u2_sim).set_index(["u1", "u2"])
        community_checkins = checkins.groupby(by=['clusterid'])
        cluster_p_nums = []
        cluster_sim = []
        for group in community_checkins:
            users = list(group[1].uid.unique())
            checkin = group[1]
            core_user = checkin[checkin.is_core == 1].uid.values[0]
            users.remove(core_user)
            users.insert(0, core_user)
            clusterId = group[1].clusterid.values[0]
            result = self.cal_l_feature(users, q, clusterId)
            result1 = self.cal_community_similarity(users, m, clusterId)
            cluster_p_nums.append(result)
            cluster_sim.append(result1)
        cluster_p_nums = pd.DataFrame(cluster_p_nums, columns=['clusterid', 'p_nums'])
        cluster_sim = pd.DataFrame(cluster_sim, columns=['clusterid', 'flag'])
        cluster_p_nums_lack = cluster_p_nums[cluster_p_nums.p_nums < k]
        cluster_p_nums_over = cluster_p_nums[cluster_p_nums.p_nums > k]

        cluster_sim_lack = cluster_sim[cluster_sim.flag == 0]
        cluster_sim_lack_clusterIds = list(cluster_sim_lack.clusterid.unique())
        cluster_p_nums_lack_clusterIds = list(cluster_p_nums_lack.clusterid.unique())
        cluster_sim_lack_clusterIds.sort()
        cluster_p_nums_lack_clusterIds.sort()
        lack_checkins = deepcopy(checkins[checkins.clusterid.isin(cluster_p_nums_lack_clusterIds)])
        comu = lack_checkins.groupby(by=['clusterid'])

        # for group in comu:
        #     u1_checkins = group[1]
        #     uids = list(u1_checkins.uid.unique())
        #     clusterId = group[1].clusterid.values[0]
        #     result = self.cal_l_feature(uids, q, clusterId)
            # u1_freq_locids = pd.DataFrame(u1_checkins['locid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
            # u1_freq_locids.columns = ['locid', 'cnt']
            # u1_freq_locids = u1_freq_locids.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)

        print(len(community_checkins), len(cluster_sim_lack), len(cluster_p_nums_lack), len(cluster_p_nums_over))
        file = open("G:/pyfile/relation_protect/src/data/result_data/community_result.txt", 'a', encoding='UTF-8')
        file.write(str(k) + '\n' + str(len(community_checkins)) + ' ' + str(len(cluster_sim_lack))+' '+ str(len(cluster_p_nums_over)) +
                   ' ' + str(len(cluster_p_nums_lack)) + '\n')
        file.write('相似性'+str(cluster_sim_lack_clusterIds) + '\n' + '多样性' + str(cluster_p_nums_lack_clusterIds) + '\n')
        file.write('交集：'+ str(len(set(cluster_sim_lack_clusterIds).intersection(set(cluster_p_nums_lack_clusterIds))))+ '\n')
        file.close()
        unionlist = list(set(cluster_sim_lack_clusterIds).union(set(cluster_p_nums_lack_clusterIds)))
        return cluster_p_nums[cluster_p_nums.clusterid.isin(unionlist)]



if __name__ == "__main__":
    start = time.time()
    # for i in [0.35, 0.3, 0.2, 0.19, 0.15, 0.13, 0.12, 0.1]:
    # for k in [3, 4, 5, 6, 7, 8, 9, 10]:
    for k in [3]:
        start1 = time.time()
        # test = cal_l_fetaure("G:/pyfile/relation_protect/src/data/city_data/", "1")
        # test.test(k, 1/3, 0.54)
        test = cal_l_fetaures("G:/pyfile/relation_protect/src/data/result_data/", "SNAP_NY_1")
        test.test(k, math.exp(-0.8), math.exp(-1/3))  # q,m分别为多样性和相似度
        # test.test(k, math.exp(-1/3), math.exp(-1))
        print(str(time.time()-start1))
    end = time.time()
    print(str(end-start))