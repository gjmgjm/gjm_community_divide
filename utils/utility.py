#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: jaculamata 
@license: Apache Licence  
@contact: 819436557@qq.com 
@site: http://blog.csdn.net/hqzxsc2006 
@software: PyCharm 
@file: 6_utility.py 
@time: 2019/1/20 15:44 
"""
import pandas as pd
from numpy.linalg import norm
from scipy.stats import entropy
from cal_similarity import cal_similarity
from copy import deepcopy
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M, base=2) + entropy(_Q, _M, base=2))


class utility():
    def __init__(self):
        pass

    def set_checkins(self, checkins, checkins_obf, pairs_path, result_pair_path, city, city_result, method):
        self.checkins = checkins
        self.checkins_obf = checkins_obf
        self.pairs_path = pairs_path
        self.result_pair_path = result_pair_path
        self.method = method
        self.city = city       # 原始数据相似度计算文件路径
        self.city_result = city_result  # 保护后数据相似度计算路径文件
        users = self.checkins_obf.uid.unique().tolist()
        users.sort(reverse=False)
        # print(users)
        self.users = np.array(deepcopy(users))  # 将用户的id升序排列


    def checkin_sim_list1(self):
        checkin_cal_sim = cal_similarity()
        checkin_cal_sim.set_checkins(self.checkins.values.tolist(), self.city_result, 30, 40, [30.387953, -97.843911, 30.249935, -97.635460])
        self.checkin_sim_list = checkin_cal_sim.cal_user_pairs()
        self.checkin_sim_list.to_csv(self.result_pair_path + self.city_result + "_" + self.method + "_after.similarity", index=False, header=False)
        print("保护后数据相似度计算完成")

    def checkin_sim_list2(self):
        checkin_cal_sim = comloc_similarity()
        checkin_cal_sim.set_checkins(self.checkins, self.city_result)
        self.checkin_sim_list = checkin_cal_sim.cal_user_pairs()
        self.checkin_sim_list.to_csv(self.result_pair_path + self.city_result + "_" + self.method + "_after.similarity", index=False, header=False)
        print("保护后数据相似度计算完成")

    def run(self, u1_checkins, u1_checkins_obf, u):
        comloc_nums = []
        comloc_nums_obf = []
        u1_locids = u1_checkins.locid.unique()
        u1_locids_obf = u1_checkins_obf.locid.unique()
        list(map(lambda x: comloc_nums.append(len(set(u1_locids).intersection(set(self.checkins[self.checkins.uid == x].locid.unique())))), self.users))
        list(map(lambda x: comloc_nums_obf.append(len(set(u1_locids_obf).intersection(set(self.checkins_obf[self.checkins_obf.uid == x].locid.unique())))), self.users))
        comloc_sum = sum(comloc_nums)
        comloc_nums = [num / comloc_sum for num in comloc_nums]
        comloc_sum_obf = sum(comloc_nums_obf)
        comloc_nums_obf = [num / comloc_sum_obf for num in comloc_nums_obf]
        sim_utility = JSD(np.array(comloc_nums), np.array(comloc_nums_obf))
        return [u, sim_utility]

    def sim_utility2(self):
        # checkin = self.checkins
        # checkin_obf = self.checkins_obf
        # sim_utility = 0.0
        core_num = multiprocessing.cpu_count()
        sim = Parallel(n_jobs=core_num)(delayed(self.run)(self.checkins[self.checkins.uid == u],
                                                         self.checkins_obf[self.checkins_obf.uid == u], u)
                                        for u in self.users)
        sim = pd.DataFrame(sim, columns=['u', 'similarity'])
        avg_sim = np.mean(sim.similarity)
        # for u in checkin_obf.uid.unique():
        #     u1_checkins = checkin[checkin.uid == u]
        #     u1_checkins_obf = checkin_obf[checkin_obf.uid == u]
        #     comloc_nums = []
        #     comloc_nums_obf = []
        #     u1_locids = u1_checkins.locid.unique()
        #     u1_locids_obf = u1_checkins_obf.locid.unique()
        #     list(map(lambda x: comloc_nums.append(len(set(u1_locids).intersection(set(checkin[checkin.uid == x].locid.unique())))), self.users))
        #     list(map(lambda x: comloc_nums_obf.append(len(set(u1_locids_obf).intersection(set(checkin_obf[checkin_obf.uid == x].locid.unique())))), self.users))
        #     comloc_sum = sum(comloc_nums)
        #     comloc_nums = [num/comloc_sum for num in comloc_nums]
        #     comloc_sum_obf = sum(comloc_nums_obf)
        #     comloc_nums_obf = [num / comloc_sum_obf for num in comloc_nums_obf]
        #     sim_utility += JSD(np.array(comloc_nums), np.array(comloc_nums_obf))
        # print("f:", sim_utility / len(checkin_obf.uid.unique()))
        # return sim_utility / len(checkin_obf.uid.unique())
        return avg_sim


    def sim_utility(self, a, k):
        print("保护前后位置访问频率相近用户改变")
        num = 0
        # 保护前后位置访问频率相似度
        self.checkin_obf_sim_list = pd.read_csv(self.pairs_path + self.city + "_" + self.method + ".similarity", names=["u1", "u2", "similarity"], header=None)
        print("保护前相似度文件读取成功")
        self.checkin_sim_list = pd.read_csv(self.result_pair_path + self.city_result + "_" + self.method + "_after.similarity", names=["u1", "u2", "similarity"], header=None)
        checkin_cnt = 0
        checkin_chunk_size = int(len(self.checkins_obf.uid.unique()) / 10)
        u_sim = 0.0
        for u in self.checkins_obf.uid.unique():
            if checkin_cnt % checkin_chunk_size == 0:  # finished the anonymization of a chunk of checkins打印一部分匿名化的结果
                print('%-3d%% work complete.' % (int(checkin_cnt / checkin_chunk_size) * 10))
            u_checkin = self.checkin_sim_list.loc[((self.checkin_sim_list.u1 == u) | (self.checkin_sim_list.u2 == u))]
            u_checkin_obf = self.checkin_obf_sim_list.loc[
                ((self.checkin_obf_sim_list.u1 == u) | (self.checkin_obf_sim_list.u2 == u))]
            u_checkin = u_checkin.sort_values(by='similarity', ascending=False).reset_index(drop=True)
            u_checkin_obf = u_checkin_obf.sort_values(by='similarity', ascending=False).reset_index(drop=True)
            u_checkin_a_sim = u_checkin[u_checkin.similarity >= a]
            u_checkin_obf_a_sim = u_checkin_obf[u_checkin_obf.similarity >= a]
            if len(u_checkin_obf_a_sim) == 0:
                if len(u_checkin_a_sim) == 0:
                    num += 1
                    u_checkin_obf_a_sim = u_checkin_obf[0:(k-1)]
                    u_checkin_a_sim = u_checkin[0:(k-1)]
            u_checkin_a_sim_set = set.union(set(u_checkin_a_sim['u1'].values), set(u_checkin_a_sim['u2'].values))
            u_checkin_obf_a_sim_set = set.union(set(u_checkin_obf_a_sim['u1'].values), set(u_checkin_obf_a_sim['u2'].values))
            list1 = list(u_checkin_a_sim_set.intersection(u_checkin_obf_a_sim_set))
            if u in list1:
                list1.remove(u)
            if len(u_checkin_obf_a_sim) == 0:
                u_sim += 0
            else:
                u_sim += len(list1) * 1.0 / len(u_checkin_obf_a_sim)
            checkin_cnt += 1
        utility = u_sim / len(self.checkins_obf.uid.unique())
        print("sim_utility:", utility)
        print("num", num)
        return utility

    def sim_utility1(self, a):
        print("保护前后位置访问频率相近用户改变")
        # 保护后位置访问频率相似度
        self.checkin_obf_sim_list = pd.read_csv(self.pairs_path + self.city + "_comloc.similarity", names=["u1", "u2", "similarity"], header=None)
        print("保护前相似度文件读取成功")
        self.checkin_sim_list = pd.read_csv(self.result_pair_path + self.city_result + "_" + self.method + "_after.similarity", names=["u1", "u2", "similarity"], header=None)
        checkin_cnt = 0
        checkin_chunk_size = int(len(self.checkins_obf.uid.unique()) / 10)
        u_sim = 0.0
        for u in self.checkins_obf.uid.unique():
            if checkin_cnt % checkin_chunk_size == 0:  # finished the anonymization of a chunk of checkins打印一部分匿名化的结果
                print('%-3d%% work complete.' % (int(checkin_cnt / checkin_chunk_size) * 10))
            u_checkin = self.checkin_sim_list.loc[((self.checkin_sim_list.u1 == u) | (self.checkin_sim_list.u2 == u))]
            u_checkin_obf = self.checkin_obf_sim_list.loc[
                ((self.checkin_obf_sim_list.u1 == u) | (self.checkin_obf_sim_list.u2 == u))]
            u_checkin = u_checkin.sort_values(by='similarity', ascending=False).reset_index(drop=True)
            u_checkin_obf = u_checkin_obf.sort_values(by='similarity', ascending=False).reset_index(drop=True)
            u_checkin_a_sim = u_checkin[u_checkin.similarity >= a]
            u_checkin_obf_a_sim = u_checkin_obf[u_checkin_obf.similarity >= a]
            if len(u_checkin_obf_a_sim) == 0:
                if len(u_checkin_a_sim) == 0:
                    u_checkin_obf_a_sim = u_checkin_obf[0:8]
                    u_checkin_a_sim = u_checkin[0:8]
            u_checkin_a_sim_set = set.union(set(u_checkin_a_sim['u1'].values), set(u_checkin_a_sim['u2'].values))
            u_checkin_obf_a_sim_set = set.union(set(u_checkin_obf_a_sim['u1'].values), set(u_checkin_obf_a_sim['u2'].values))
            list1 = list(u_checkin_a_sim_set.intersection(u_checkin_obf_a_sim_set))
            if u in list1:
                list1.remove(u)
            if len(u_checkin_obf_a_sim) == 0:
                u_sim += 0
            else:
                u_sim += len(list1) * 1.0 / len(u_checkin_obf_a_sim)
            checkin_cnt += 1
        utility = u_sim / len(self.checkins_obf.uid.unique())
        print("sim_utility:", utility)
        return utility

    # def community_utility(self, a, k):
    #     print("保护前后位置访问频率相近用户改变")
    #     # self.checkin_sim_list1()
    #     # 社区划分位置访问频率相似度
    #     self.checkin_obf_sim_list = pd.read_csv(self.pairs_path + self.city + "_freqscatter.similarity", names=["u1", "u2", "similarity"], header=None)
    #     self.checkin_sim_list = pd.read_csv(self.result_pair_path + self.city_result + "_" + self.method + "_after.similarity", names=["u1", "u2", "similarity"], header=None)
    #     checkin_cnt = 0
    #     checkin_chunk_size = int(len(self.checkins_obf.uid.unique()) / 10)
    #     u_sim = 0.0
    #     for u in self.checkins_obf.uid.unique():
    #         if checkin_cnt % checkin_chunk_size == 0:  # finished the anonymization of a chunk of checkins打印一部分匿名化的结果
    #             print('%-3d%% work complete.' % (int(checkin_cnt / checkin_chunk_size) * 10))
    #         u_checkin = self.checkin_sim_list.loc[((self.checkin_sim_list.u1 == u) | (self.checkin_sim_list.u2 == u))]
    #         u_checkin_obf = self.checkin_obf_sim_list.loc[
    #             ((self.checkin_obf_sim_list.u1 == u) | (self.checkin_obf_sim_list.u2 == u))]
    #         u_checkin = u_checkin.sort_values(by='similarity', ascending=False).reset_index(drop=True)
    #         u_checkin_obf = u_checkin_obf.sort_values(by='similarity', ascending=False).reset_index(drop=True)
    #         u_checkin_a_sim = u_checkin[u_checkin.similarity >= a]
    #         u_checkin_obf_a_sim = u_checkin_obf[u_checkin_obf.similarity >= a]
    #         if len(u_checkin_obf_a_sim) < k:  # 对于不满足l邻居的用户
    #             if len(u_checkin_obf_a_sim) == 0:  # 没有邻居的用户,改变最大，与原始数据是没有交集的
    #                 u_sim += 0
    #                 checkin_cnt += 1
    #                 continue
    #             else:
    #                 u_checkin_a_sim = u_checkin[0:k]  # 进行保护之后用户u的邻居
    #         else:                                     # 满足l邻居的用户,邻居应该取与原始数据相同的长度
    #             u_checkin_a_sim = u_checkin[0:len(u_checkin_obf_a_sim)]
    #         u_checkin_a_sim_set = set.union(set(u_checkin_a_sim['u1'].values), set(u_checkin_a_sim['u2'].values))
    #         u_checkin_obf_a_sim_set = set.union(set(u_checkin_obf_a_sim['u1'].values),
    #                                             set(u_checkin_obf_a_sim['u2'].values))
    #         list1 = list(u_checkin_a_sim_set.intersection(u_checkin_obf_a_sim_set))
    #         list1.remove(u)
    #         u_sim += len(list1) * 1.0 / len(u_checkin_obf_a_sim)
    #         checkin_cnt += 1
    #     utility = u_sim / len(self.checkins_obf.uid.unique())
    #     print("sim_utility:", utility)
    #     return utility


