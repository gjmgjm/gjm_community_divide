#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: jaculamata 
@license: Apache Licence  
@contact: 819436557@qq.com 
@site: http://blog.csdn.net/hqzxsc2006 
@software: PyCharm 
@file: 5_security.py 
@time: 2019/1/20 15:44 
"""
import pandas as pd
from numpy.linalg import norm
from scipy.stats import entropy
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from scipy.special import comb
import time
import math
from itertools import combinations
from copy import deepcopy
from grid_divide import grid_divide



def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M, base=2) + entropy(_Q, _M, base=2))


def single_security(u1_locids, u2_locids, u1_locids_obf, u2_locids_obf, i):
    # print(i)
    u1_u2_com_locids = set(u1_locids).intersection(set(u2_locids))
    u1_u2_com_locids_obf = set(u1_locids_obf).intersection(set(u2_locids_obf))
    u1_u2_com = list(u1_u2_com_locids.intersection(u1_u2_com_locids_obf))
    return len(u1_u2_com) * 1.0 / len(u1_u2_com_locids_obf)


def single_security1(u1_checkin, u2_checkin, u1_checkin_obf, u2_checkin_obf):
    # print(i)
    u1_locids = u1_checkin.locid.unique()
    u2_locids = u2_checkin.locid.unique()
    u1_locids_obf = u1_checkin_obf.locid.unique()
    u2_locids_obf = u2_checkin_obf.locid.unique()
    u1_u2_com_locids = set(u1_locids).intersection(set(u2_locids))
    u1_u2_com_locids_obf = set(u1_locids_obf).intersection(set(u2_locids_obf))
    u1_u2_com = list(u1_u2_com_locids.intersection(u1_u2_com_locids_obf))
    if len(u1_u2_com_locids_obf) == 0:
        return 0
    return len(u1_u2_com) * 1.0 / len(u1_u2_com_locids_obf)


class security_unility():

    def __init__(self, lons_per_km, lats_per_km):
        # self.lons_per_km = 0.005202 * 2  # delta longitudes per kilo meter AS
        # self.lons_per_km = 0.005681 * 2  # SF
        # self.lats_per_km = 0.004492 * 2  # delta latitudes per kilo meter
        # self.lons_per_km = 0.0059352 * 2  # NY
        # self.lats_per_km = 0.0044966 * 2  # delta latitudes per kilo meter
        self.lons_per_km = lons_per_km
        self.lats_per_km = lats_per_km
        pass

    def set_checkins(self, checkins, checkins_obf, pairs_path, city):
        self.checkins = checkins
        self.checkins_obf = checkins_obf
        self.pairs_path = pairs_path
        self.city = city
        self.uid_pairs_obf = []  # 用户对
        self.uid_pairs_fres_obf = []  # 用户对产生的个数
        # print("保护前后数据读取成功")

    def set_grid(self, m, n, range):
        self.m = m
        self.n = n
        self.range = range

    # 两点之间的欧氏距离计算
    def euclidean_distance(self, loc1, loc2):
        return math.sqrt(((loc1[1] - loc2[1]) / self.lons_per_km) ** 2 + ((loc1[0] - loc2[0]) / self.lats_per_km) ** 2)

    def unility(self):
        print()
        print("用户间共同访问位置改变情况")
        similarity_pairs = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/" + self.city + "_comloc.similarity",names=["u1","u2","similarity"])
        pairs = similarity_pairs[similarity_pairs.similarity > 0]
        users = list(self.checkins.uid.unique())
        pairs = pairs.ix[:, [0, 1]]
        pairs = pairs[pairs.u1.isin(users)]
        pairs = pairs[pairs.u2.isin(users)]
        print("有共同访问位置的用户对数：", len(pairs))
        core_num = multiprocessing.cpu_count()
        meet_cell = Parallel(n_jobs=core_num)(delayed(single_security)(
            self.checkins.loc[self.checkins.uid == pairs.iloc[i]['u1']].locid.unique(),
            self.checkins.loc[self.checkins.uid == pairs.iloc[i]['u2']].locid.unique(),
            self.checkins_obf.loc[self.checkins_obf.uid == pairs.iloc[i]['u1']].locid.unique(),
            self.checkins_obf.loc[self.checkins_obf.uid == pairs.iloc[i]['u2']].locid.unique(), i
        ) for i in range(len(pairs)))
        security = sum(meet_cell) / len(pairs)
        print("a:",security)
        return security


    # def a_security(self):
    #     print()
    #     print("用户间共同访问位置改变情况")
    #     locids_uids = self.checkins.groupby(by=['locid', 'uid']).size().reset_index(name='uid_nums')
    #     uid_pairs = []  # 用户对
    #     uid_pairs_fres = []  # 用户对产生的个数
    #
    #     checkin_cnt = 0
    #     checkin_chunk_size = int(len(self.checkins_obf.locid.unique()) / 10)
    #     # 剩下的进行统计
    #     for locid in self.checkins_obf.locid.unique():
    #         if checkin_cnt % checkin_chunk_size == 0:  # finished the anonymization of a chunk of checkins打印一部分匿名化的结果
    #             print('%-3d%% work complete.' % (int(checkin_cnt / checkin_chunk_size) * 10))
    #         locid_uids = locids_uids[locids_uids.locid == locid].uid.values
    #         if len(locid_uids) > 1:       # 保护后用户对
    #             locid_uid_pairs = list(combinations(locid_uids, 2))
    #             for i in locid_uid_pairs:
    #                 if set(i) in uid_pairs:
    #                     uid_pairs_fres[uid_pairs.index(set(i))] += 1
    #                 else:
    #                     uid_pairs.append(set(i))
    #                     uid_pairs_fres.append(1)
    #         checkin_cnt += 1
    #     uid_pairs_fres_set = set(uid_pairs_fres)
    #     uid_pairs_freqs = list(map(lambda x: uid_pairs_fres.count(x), uid_pairs_fres_set))
    #     freqs_sum = sum(uid_pairs_freqs)
    #     uid_pairs_freqs = [freq / freqs_sum for freq in uid_pairs_freqs]
    #     print("a_security:", uid_pairs_freqs)
    #     q1 = entropy(uid_pairs_freqs)
    #     print(q1)
    #     return uid_pairs_freqs

    def a_security(self):
        # start = time.time()
        print()
        print("用户间共同访问位置改变情况")
        users = list(self.checkins_obf.uid.unique())
        users.sort(reverse=False)
        user = np.array(deepcopy(users))  # 将用户的id进行从小到大的排序
        pairs = list(combinations(user, 2))
        pairs = pd.DataFrame(pairs, columns=['u1', 'u2'])
        core_num = multiprocessing.cpu_count()
        meet_cell = Parallel(n_jobs=core_num)(delayed(single_security)(
            self.checkins.loc[self.checkins.uid == pairs.iloc[i]['u1']].locid.unique(),
            self.checkins.loc[self.checkins.uid == pairs.iloc[i]['u2']].locid.unique(),
            self.checkins_obf.loc[self.checkins_obf.uid == pairs.iloc[i]['u1']].locid.unique(),
            self.checkins_obf.loc[self.checkins_obf.uid == pairs.iloc[i]['u2']].locid.unique(),i
        ) for i in range(len(pairs)))
        security = sum(meet_cell) / comb(len(users), 2)
        # end = time.time()
        # print("总共花的费时间为", str(end - start))
        print("a:", security)
        return security

    def a_security1(self, checkins):

        checkins_obf = self.checkins_obf[self.checkins_obf.uid.isin(list(checkins.uid.unique()))]
        users = list(checkins_obf.uid.unique())
        users.sort(reverse=False)
        user = np.array(deepcopy(users))  # 将用户的id进行从小到大的排序
        pairs = list(combinations(user, 2))
        pairs = pd.DataFrame(pairs, columns=['u1', 'u2'])
        core_num = multiprocessing.cpu_count()
        meet_cell = Parallel(n_jobs=core_num)(delayed(single_security)(
            checkins.loc[checkins.uid == pairs.iloc[i]['u1']].locid.unique(),
            checkins.loc[checkins.uid == pairs.iloc[i]['u2']].locid.unique(),
            checkins_obf.loc[checkins_obf.uid == pairs.iloc[i]['u1']].locid.unique(),
            checkins_obf.loc[checkins_obf.uid == pairs.iloc[i]['u2']].locid.unique()
        ) for i in range(len(pairs)))
        security = sum(meet_cell) / comb(len(users), 2)
        return security

    def a_security2(self, checkins):

        checkins_obf = self.checkins_obf[self.checkins_obf.uid.isin(list(checkins.uid.unique()))]
        users = list(checkins_obf.uid.unique())
        users.sort(reverse=False)
        user = np.array(deepcopy(users))  # 将用户的id进行从小到大的排序
        pairs = list(combinations(user, 2))
        pairs = pd.DataFrame(pairs, columns=['u1', 'u2'])
        core_num = multiprocessing.cpu_count()
        meet_cell = Parallel(n_jobs=core_num)(delayed(single_security1)(
            checkins.loc[checkins.uid == pairs.iloc[i]['u1']],
            checkins.loc[checkins.uid == pairs.iloc[i]['u2']],
            checkins_obf.loc[checkins_obf.uid == pairs.iloc[i]['u1']],
            checkins_obf.loc[checkins_obf.uid == pairs.iloc[i]['u2']]
        ) for i in range(len(pairs)))
        security = sum(meet_cell) / comb(len(users), 2)
        return security

    def all_a_security(self):
        community_checkins = self.checkins.groupby(by=['clusterid'])
        security = 0
        for group in community_checkins:
            security += self.a_security1(group[1])
        security = security/len(community_checkins)
        return security

    def all_a_security1(self):
        community_checkins = self.checkins.groupby(by=['clusterid'])
        security = 0
        for group in community_checkins:
            security += self.a_security2(group[1])
        security = security / len(community_checkins)
        return security

    def b_security(self, m):
        print()
        print("用户频繁访问位置改变情况")
        # start = time.time()
        checkin = self.checkins
        checkin_obf = self.checkins_obf
        single_security = 0
        checkins_obf_uids = checkin_obf.uid.unique()
        for u in checkin_obf.uid.unique():
            u_checkin = checkin.loc[checkin.uid == u]
            u_checkin_obf = checkin_obf.loc[checkin_obf.uid == u]
            u_loc_distr = pd.DataFrame(u_checkin['locid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
            u_loc_distr.columns = ['locid', 'cnt']
            u_loc_distr = u_loc_distr.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
            u_loc_distr_obf = pd.DataFrame(u_checkin_obf['locid'].value_counts()).reset_index()
            u_loc_distr_obf.columns = ['locid', 'cnt']
            u_loc_distr_obf = u_loc_distr_obf.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
            if m <= len(u_loc_distr):
                u_loc_distr = u_loc_distr[0:m]
            if m <= len(u_loc_distr_obf):
                u_loc_distr_obf = u_loc_distr_obf[0:m]
            itstlist = list(set(u_loc_distr['locid'].values).intersection(set(u_loc_distr_obf['locid'].values)))
            single_security += len(itstlist) * 1.0/len(u_loc_distr_obf)
        security = single_security/len(checkins_obf_uids)
        # end = time.time()
        # print("总共花的费时间为", str(end - start))
        print("b:", security)
        return security

    def c_security(self, m):
        # print()
        print("全局的频繁访问位置改变情况")
        # start = time.time()
        checkin = self.checkins
        checkin_obf = self.checkins_obf
        checkin_fre = pd.DataFrame(checkin['locid'].value_counts()).reset_index()
        checkin_fre.columns = ['locid', 'cnt']
        checkin_fre = checkin_fre.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
        checkin_obf_fre = pd.DataFrame(checkin_obf['locid'].value_counts()).reset_index()
        checkin_obf_fre.columns = ['locid', 'cnt']
        checkin_obf_fre = checkin_obf_fre.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
        checkin_fre = checkin_fre[0:m]
        checkin_obf_fre = checkin_obf_fre[0:m]
        itstlist = list(set(checkin_fre['locid'].values).intersection(set(checkin_obf_fre['locid'].values)))
        globle_security = len(itstlist) * 1.0 / len(checkin_obf_fre)
        # end = time.time()
        # print("总共花的费时间为", str(end - start))
        print("c:", globle_security)
        return globle_security

    def d_security(self):
        print()
        print("全局位置访问频率分布改变情况")   # 全局位置访问频率分布改变情况，保护前后访问频率分布分别为Ta,Tb
        # start = time.time()
        checkin = self.checkins
        checkin_obf = self.checkins_obf
        checkin_len = len(checkin)
        checkin_obf_len = len(checkin_obf)
        union_grid_id = list(set(checkin.locid.unique()).union(set(checkin_obf.locid.unique())))
        checkin_vec = list(map((lambda x: len(checkin[checkin.locid == x]) * 1.0 / checkin_len), union_grid_id))
        checkin_obf_vec = list(map((lambda x: len(checkin_obf[checkin_obf.locid == x]) * 1.0 / checkin_obf_len), union_grid_id))
        checkin_vec = np.array(list(checkin_vec))
        checkin_obf_vec = np.array(list(checkin_obf_vec))
        globle_security = JSD(checkin_vec, checkin_obf_vec)
        print("d:", globle_security)
        return globle_security

    def e_security(self):
        print()
        print("用户位置访问频率分布改变情况")
        checkin = self.checkins
        checkin_obf = self.checkins_obf
        single_security = 0
        for u in checkin_obf.uid.unique():
            u_checkin = checkin.loc[checkin.uid == u]
            u_checkin_obf = checkin_obf.loc[checkin_obf.uid == u]
            u_checkin_len = len(u_checkin)
            u_checkin_obf_len = len(u_checkin_obf)
            union_grid_id = list(set(u_checkin.locid.unique()).union(set(u_checkin_obf.locid.unique())))
            checkin_vec = list(map((lambda x: len(u_checkin[u_checkin.locid == x]) * 1.0 / u_checkin_len), union_grid_id))
            checkin_obf_vec = list(map((lambda x: len(u_checkin_obf[u_checkin_obf.locid == x]) * 1.0 / u_checkin_obf_len), union_grid_id))
            single_security += JSD(np.array(checkin_vec), np.array(checkin_obf_vec))
        print("e:", single_security/len(checkin_obf.uid.unique()))
        return single_security/len(checkin_obf.uid.unique())

    def f_security(self):
        print()
        print("位置改变个数统计")
        nums = 0
        for row in self.checkins.itertuples(index=False, name=False):
            if row[5] != row[8]:
                nums += 1
            if (row[5] == row[8]) & (row[5] == -1):
                nums += 1
        print("f:", nums)
        return nums

    def f_security1(self, k):    # k-匿名改变的位置个数
        print()
        print("位置改变个数统计")
        grid_divider = grid_divide(self.checkins.values.tolist(), self.n, self.m, self.range)
        checkins = grid_divider.divide_area_by_NN().ix[:, [0, 1, 2, 3, 4, 5]].reset_index(drop=True)  # 带有网格id
        nums = 0
        for i in range(0, len(checkins), k):
            if i == len(checkins):
                break
            checkin = deepcopy(checkins[i:(i + k)])
            grid_ids = [row[5] for row in checkin.itertuples(index=False, name=False)]
            core_grid = grid_ids[0]
            del grid_ids[0]
            for grid in grid_ids:
                if grid != core_grid:
                    nums += 1
        print("f:", nums)
        return nums

    def f_security2(self):   # 随机扰动变化的位置个数
        print()
        print("位置改变个数统计")
        grid_divider = grid_divide(self.checkins.values.tolist(), self.n, self.m, self.range)
        checkins = grid_divider.divide_area_by_NN().ix[:, [0, 1, 2, 3, 4, 5]].reset_index(drop=True)  # 带有网格id
        grid_divider = grid_divide(self.checkins_obf.values.tolist(), self.n, self.m, self.range)
        checkins_obf = grid_divider.divide_area_by_NN().ix[:, [0, 1, 2, 3, 4, 5]].reset_index(drop=True)  # 带有网格id
        checkins.loc[:, 'grid_id_before'] = checkins_obf['grid_id']
        nums = 0
        for row in checkins.itertuples(index=False, name=False):
            if row[5] != row[6]:
                nums += 1
        print("f:", nums)
        return nums

    def g_security(self):  # 社区划分的位置距离的改变,要分社区了
        print()
        from get_core_point import get_center_point
        core_users = self.checkins[self.checkins.is_core == 1].uid.unique()
        get_point = get_center_point(self.checkins_obf)
        center_point = get_point.user_core_point(core_users)
        # print("社区核心用户的中心位置计算完成")
        distance = 0
        community_checkins = self.checkins.groupby(by=['clusterid'])
        checkin1 = self.checkins_obf.groupby(by=['locid', 'latitude', 'longitude']).size().reset_index( name="locid_time")
        locids = [row[0] for row in checkin1.itertuples(index=False, name=False)]  # 记录locid对应的经纬度，以便在替换locid时将相应的位置数据也进行替换
        lat_lon = [[row[1], row[2]] for row in checkin1.itertuples(index=False, name=False)]
        # print(len(community_checkins))
        for group in community_checkins:
            core_user = group[1][group[1].is_core == 1].uid.values[0]
            u_center_loc = center_point[center_point.uid == core_user].values.tolist()[0]
            loc_center = [u_center_loc[1], u_center_loc[2]]
            for row in group[1].itertuples(index=False, name=False):
                if (row[5] == row[8]) & (row[5] == -1):  # row[9]是原来的locid，row[4]是位置替换后的locid
                    distance += self.euclidean_distance(loc_center, [row[2], row[3]])
                if row[4] != row[9]:
                    distance += self.euclidean_distance(lat_lon[locids.index(row[9])], [row[2], row[3]])
        print("位置距离改变:", distance)
        return distance

    def g_security1(self, k):  # dls方案的替换位置的距离改变,其中k为匿名的个数
        print()
        distance = 0
        for i in range(0, len(self.checkins), k):
            if i == len(self.checkins):
                break
            checkins = deepcopy(self.checkins[i:(i+k)])
            location = [[row[2], row[3]] for row in checkins.itertuples(index=False, name=False)]
            core_loc = location[0]
            del location[0]
            for loc in location:
                distance += self.euclidean_distance(core_loc, loc)
        print("位置距离改变:", distance)
        return distance

    def g_security2(self):  # 随机扰动的位置距离改变
        print()
        # from get_core_point import get_center_point
        # get_point = get_center_point(self.checkins_obf)
        # users = self.checkins_obf.uid.unique()
        # center_point = get_point.user_core_point(users)
        checkins = deepcopy(self.checkins)
        checkins.loc[:, 'locid_before'] = self.checkins_obf['locid']
        checkin1 = self.checkins_obf.groupby(by=['locid', 'latitude', 'longitude']).size().reset_index(name="locid_time")
        locids = [row[0] for row in checkin1.itertuples(index=False, name=False)]  # 记录locid对应的经纬度，以便在替换locid时将相应的位置数据也进行替换
        lat_lon = [[row[1], row[2]] for row in checkin1.itertuples(index=False, name=False)]
        distance = 0
        for row in checkins.itertuples(index=False, name=False):
            if row[4] != row[5]:   # 说明进行了位置替换
                # u_center_loc = center_point[center_point.uid == row[0]].values.tolist()[0]
                # loc_center = [u_center_loc[1], u_center_loc[2]]
                distance += self.euclidean_distance(lat_lon[locids.index(row[5])], [row[2], row[3]])
                # distance += self.euclidean_distance(loc_center, [row[2], row[3]])
        print("位置距离改变:", distance)
        return distance


    # def entropy(self, probabilities):
    #     return -reduce(add, map(lambda x: math.log2(x) * x, probabilities))