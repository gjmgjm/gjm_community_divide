#!/usr/bin/env python
# encoding: utf-8

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from itertools import combinations
from grid_divide import grid_divide
from copy import deepcopy
import time
import gc
import numpy as np
import math
import decimal


def freqdistbt_similarity(u1_vector, u2_vector, u1, u2):  # 位置访问频率分布相似度
    u1_u2_similarity = cosine_similarity([u1_vector], [u2_vector])
    return [u1, u2, u1_u2_similarity[0][0]]

# def freqloc_similarity(u1_checkins, u2_checkins, u1, u2, k, i):  # 频繁访问位置相似度
#     # u1_freq_locids = u1_checkins.groupby(by=['latitude', 'longitude']).size().reset_index(name='times')
#     # u1_freq_locids = u1_freq_locids.sort_values(by=['times'], ascending=False).reset_index(drop=True)
#     # u2_freq_locids = u2_checkins.groupby(by=['latitude', 'longitude']).size().reset_index(name='times')  # 统计用户u2的locid及访问次数
#     # u2_freq_locids = u2_freq_locids.sort_values(by=['times'], ascending=False).reset_index(drop=True)
#     # print(i)
#     u1_freq_locids =pd.DataFrame(u1_checkins['locid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
#     u1_freq_locids.columns = ['locid', 'cnt']
#     u2_freq_locids = pd.DataFrame(u2_checkins['locid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
#     u2_freq_locids.columns = ['locid', 'cnt']
#     if k < len(u1_freq_locids):
#         u1_freq_locids = u1_freq_locids[0:k]
#     if k < len(u2_freq_locids):
#         u2_freq_locids = u2_freq_locids[0:k]
#     itstlistlen = (len(list(set(u1_freq_locids['locid'].values).intersection(set(u2_freq_locids['locid'].values))))*2)/(len(set(u1_freq_locids['locid'].values))+len(set(u2_freq_locids['locid'].values)))
#     return [u1, u2, itstlistlen]

# def comloc_similarity(u1_vector, u2_vector, u1, u2):
#     u1_u2_similarity = cosine_similarity([u1_vector], [u2_vector])
#     return [u1, u2, u1_u2_similarity[0][0]]


def comloc_similarity(u1_locids, u2_locids, u1, u2, i):
    # u1_locids = u1_checkins.locid.unique()
    # u2_locids = u2_checkins.locid.unique()
    print(i)
    comlocs = len(set(u1_locids).intersection(set(u2_locids)))
    if comlocs == 0:
        return [u1, u2, 0]
    return [u1, u2, math.exp(-1/comlocs)]



def cal_delta(x):
    circle_len = 2 * 6371.004 * math.pi
    circle_len_1 = circle_len * math.cos(x * math.pi / 180)
    delta_lng = 360 / circle_len_1
    delta_lat = 360 / (2 * 6371.004 * math.pi)  #
    print(delta_lat / 2, delta_lng / 2)  # 对应500m有多少度


def weighted_similarity(u1_checkins, u2_checkins, u1, u2, k, a, b, u1_vector, u2_vector):
    u1_freq_locids = pd.DataFrame(u1_checkins['locid'].value_counts()).reset_index().rename(columns={'locid':'cnt', 'index':'locid'})  # 统计用户u1的locid及访问次数
    # u1_freq_locids.columns = ['locid', 'cnt']
    u2_freq_locids = pd.DataFrame(u2_checkins['locid'].value_counts()).reset_index().rename(columns={'locid':'cnt', 'index':'locid'})  # 统计用户u2的locid及访问次数
    # u2_freq_locids.columns = ['locid', 'cnt']
    if k < len(u1_freq_locids):
        u1_freq_locids = u1_freq_locids[0:k]
    if k < len(u2_freq_locids):
        u2_freq_locids = u2_freq_locids[0:k]
    itstlistlen = (len(list(set(u1_freq_locids['locid'].values).intersection(set(u2_freq_locids['locid'].values)))) * 2) / (len(set(u1_freq_locids['locid'].values)) + len(set(u2_freq_locids['locid'].values)))
    u1_u2_similarity = cosine_similarity([u1_vector], [u2_vector])
    similarity = a * itstlistlen + b * u1_u2_similarity[0][0]
    return [u1, u2, similarity]


class cal_similarity():

    def __init__(self, method):
        self.method = method
        pass

    def set_checkins(self, checkins, city, k, n=0, m=0, ranges=None):
        self.checkins = checkins  # 不需要进行网格划分，利用的特征是频繁访问位置或者共同访问位置特征
        if ranges is not None:     # 表示需要进行网格划分，利用的特征是位置访问频率分布
            grid_divider = grid_divide(deepcopy(checkins), n, m, ranges)
            checkin = grid_divider.divide_area_by_NN()
            self.checkins = checkin.ix[:, [0, 1, 2, 3, 4, 5]]
            self.grid_id = list(self.checkins.grid_id.unique())
            self.latInterval = grid_divider.get_latInterval()
            self.lngInterval = grid_divider.get_lngInterval()
            self.n = n
            self.m = m
            # self.lons_per_km = decimal.Decimal.from_float(0.005202 * 2)  # delta longitudes per kilo meter  0.005681
            # self.lons_per_km = decimal.Decimal.from_float(0.005681 * 2)  # 旧金山
            # self.lats_per_km = decimal.Decimal.from_float(0.004492 * 2)  # delta latitudes per kilo meter
            self.lons_per_km = decimal.Decimal.from_float(0.0059352 * 2)  # NY
            self.lats_per_km = decimal.Decimal.from_float(0.0044966 * 2)  # delta latitudes per kilo meter
        self.city = city
        users = self.checkins.uid.unique().tolist()
        users.sort(reverse=False)
        self.users = np.array(deepcopy(users))  # 将用户的id进行从小到大的排序
        self.pairs = list(combinations(self.users, 2))
        self.pairs = pd.DataFrame(self.pairs, columns=['u1', 'u2'])
        self.k = k
        del users
        gc.collect()

    def cal_user_vector(self, u_checkins, u):
        u_vector = list(map((lambda x: len(u_checkins[u_checkins.grid_id == x]) * 1.0 / len(u_checkins)), self.grid_id))
        return [u, u_vector]

    def distance(self, grid1, grid2):
        index_i1, index_i2 = int(grid1 / self.m), int(grid2 / self.m)
        index_j1, index_j2 = grid1 % self.m, grid2 % self.m
        i_interval = abs(index_i1-index_i2)
        j_interval = abs(index_j1-index_j2)
        return math.sqrt((i_interval*self.latInterval/self.lats_per_km)**2 + (j_interval*self.lngInterval/self.lons_per_km)**2)

    def freqloc_similarity(self, u1_checkins, u2_checkins, u1_freq_locids, u2_freq_locids, u1, u2, k, i):
        print(i)
        # u1_freq_locids = pd.DataFrame(u1_checkins['locid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
        # u1_freq_locids.columns = ['locid', 'cnt']
        u1_freq_locids = u1_freq_locids.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
        # u2_freq_locids = pd.DataFrame(u2_checkins['locid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
        # u2_freq_locids.columns = ['locid', 'cnt']
        u2_freq_locids = u2_freq_locids.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
        if k < len(u1_freq_locids):
            u1_freq_locids = u1_freq_locids[0:k]
        if k < len(u2_freq_locids):
            u2_freq_locids = u2_freq_locids[0:k]
        u1_locid_gridid = list(u1_checkins[u1_checkins.locid.isin(u1_freq_locids.locid.unique())].grid_id.unique())
        u2_locid_gridid = list(u2_checkins[u2_checkins.locid.isin(u2_freq_locids.locid.unique())].grid_id.unique())
        # 计算最小距离
        locids_distance = []
        if len(set(u1_locid_gridid).intersection(set(u2_locid_gridid))) != 0:  # 频繁访问位置中存在共同访问位置，最小距离为0
            return [u1, u2, 1.0]
        else:  # 没有共同访问位置，计算最短距离
            for grid1 in u1_locid_gridid:
                for grid2 in u2_locid_gridid:
                    locids_distance.append(self.distance(grid1, grid2))
            min_distance = min(locids_distance)
        return [u1, u2, math.exp(-min_distance)]   # 归一化距离作为相似度

    def comloc_meet(self, u1_checkins, u1):
        checkins = self.checkins
        comloc_nums = []
        u1_locids = u1_checkins.locid.unique()

        list(map(lambda x: comloc_nums.append(len(set(u1_locids).intersection(set(checkins[checkins.uid == x].locid.unique())))), self.users))
        comloc_sum = sum(comloc_nums)
        comloc_nums = [num / comloc_sum for num in comloc_nums]
        return [u1, comloc_nums]

    def cal_user_pairs(self):   # 位置访问频率分布相似度
        # 计算位置访问频率向量
        core_num = multiprocessing.cpu_count()
        loc_vector = Parallel(n_jobs=core_num)(
            delayed(self.cal_user_vector)(self.checkins[self.checkins.uid == u], u) for u in self.users)
        loc_vector = pd.DataFrame(loc_vector, columns=['uid', 'vector'])
        print("位置访问频率向量计算完成")
        # 计算用户对之间的位置访问频率相似度
        meet_cell = Parallel(n_jobs=core_num)(
            delayed(freqdistbt_similarity)(loc_vector.loc[loc_vector.uid == self.pairs.loc[i, 'u1'], 'vector'].values[0],
                                           loc_vector.loc[loc_vector.uid == self.pairs.loc[i, 'u2'], 'vector'].values[0],
                                           self.pairs.loc[i, 'u1'], self.pairs.loc[i, 'u2']) for i in range(len(self.pairs)))
        pairs_sim_list = pd.DataFrame(meet_cell, columns=['u1', 'u2', 'similarity'])
        # pairs_sim_list.to_csv("G:/pyfile/relation_protect/src/data/city_data/"+self.city+"_freqscatterloc.similarity",
        #                       index=False, header=False)
        return pairs_sim_list

    def cal_freqloc_user_pairs(self, k):    # 频繁访问位置相似度
        core_num = multiprocessing.cpu_count()
        meet_cell = Parallel(n_jobs=core_num)(   # pd.DataFrame(u1_checkins['locid'].value_counts()).reset_index()
            delayed(self.freqloc_similarity)(self.checkins[self.checkins.uid == self.pairs.iloc[i]['u1']],
                                             self.checkins[self.checkins.uid == self.pairs.iloc[i]['u2']],
                                             pd.DataFrame(self.checkins[self.checkins.uid == self.pairs.iloc[i]['u1']]['locid'].value_counts()).reset_index().rename(columns={'locid': 'cnt', 'index': 'locid'}),
                                             pd.DataFrame(self.checkins[self.checkins.uid == self.pairs.iloc[i]['u2']]['locid'].value_counts()).reset_index().rename(columns={'locid': 'cnt', 'index': 'locid'}),
                                             self.pairs.iloc[i]['u1'], self.pairs.iloc[i]['u2'], k, i) for i in range(len(self.pairs)))
        pairs_sim_list = pd.DataFrame(meet_cell, columns=['u1', 'u2', 'similarity'])
        print("频繁访问位置相似度计算完成")
        pairs_sim_list.to_csv("G:/pyfile/relation_protect/src/data/city_data/" + self.city + "_freqloc.similarity", index=False, header=False)
        # pairs_sim_list.to_csv("G:/pyfile/relation_protect/src/data/city_data/" + self.city + "_freqloc_11_"+ str(self.k) +"_after.similarity",
        #                       index=False, header=False)
        # pairs_sim_list.to_csv("G:/pyfile/relation_protect/src/data/result_data/" + self.method + "/" + self.city + "_freqloc_1_" + str(
        #     self.k) + "_after.similarity",
        #                       index=False, header=False)
        return pairs_sim_list

    def cal_comloc_user_pairs(self):
        core_num = multiprocessing.cpu_count()
        # 计算用户包括自己的其他用户的共同访问位置的比例
        # loc_v = Parallel(n_jobs=core_num)(
        #     delayed(self.comloc_similarity)(self.checkins[self.checkins.uid == u], u) for u in self.users)
        # loc_vector = pd.DataFrame(loc_v, columns=['uid', 'vector'])
        meet_cell = Parallel(n_jobs=core_num)(
            delayed(comloc_similarity)(self.checkins[self.checkins.uid == self.pairs.iloc[i]['u1']].locid.unique(),
                                       self.checkins[self.checkins.uid == self.pairs.iloc[i]['u2']].locid.unique(),
                                       self.pairs.iloc[i]['u1'], self.pairs.iloc[i]['u2'], i) for i in range(len(self.pairs)))
        pairs_sim_list = pd.DataFrame(meet_cell, columns=['u1', 'u2', 'similarity'])
        pairs_sim_list.to_csv("G:/pyfile/relation_protect/src/data/city_data/" + self.city + "_comloc.similarity", index=False, header=False)

        print("共同访问位置的比例计算完成")
        # 计算用户间共同访问位置余弦相似度
        # meet_cell = Parallel(n_jobs=core_num)(
        #     delayed(comloc_similarity)(loc_vector.loc[loc_vector.uid == self.pairs.loc[i, 'u1'], 'vector'].values[0],
        #                                 loc_vector.loc[loc_vector.uid == self.pairs.loc[i, 'u2'], 'vector'].values[0],
        #                                 self.pairs.loc[i, 'u1'], self.pairs.loc[i, 'u2']) for i in range(len(self.pairs)))
        # pairs_sim_list = pd.DataFrame(meet_cell, columns=['u1', 'u2', 'similarity'])
        # pairs_sim_list.to_csv("G:/pyfile/relation_protect/src/data/city_data/" + self.city + "_comloc_11_"+ str(self.k) +"_after.similarity",
        #                       index=False, header=False)
        # pairs_sim_list.to_csv("G:/pyfile/relation_protect/src/data/result_data/" + self.method+"/" + self.city + "_comloc_1_" + str(
        #     self.k) + "_after.similarity", index=False, header=False)
        return pairs_sim_list

    def cal_weighted_sim(self, k, a, b):
        core_num = multiprocessing.cpu_count()
        loc_vector = Parallel(n_jobs=core_num)(
            delayed(self.cal_user_vector)(self.checkins[self.checkins.uid == u], u) for u in self.users)
        loc_vector = pd.DataFrame(loc_vector, columns=['uid', 'vector'])
        print("位置访问频率向量计算完成")
        meet_cell = Parallel(n_jobs=core_num)(
            delayed(weighted_similarity)(self.checkins[self.checkins.uid == self.pairs.iloc[i]['u1']],
                                         self.checkins[self.checkins.uid == self.pairs.iloc[i]['u2']],
                                         self.pairs.iloc[i]['u1'], self.pairs.iloc[i]['u2'], k, a, b,
                                         loc_vector.loc[loc_vector.uid == self.pairs.loc[i, 'u1'], 'vector'].values[0],
                                         loc_vector.loc[loc_vector.uid == self.pairs.loc[i, 'u2'], 'vector'].values[0],
                                         ) for i in range(len(self.pairs)))
        pairs_sim_list = pd.DataFrame(meet_cell, columns=['u1', 'u2', 'similarity'])
        return pairs_sim_list


if __name__ == "__main__":

    # 37.809524, -122.520352, 37.708991, -122.358712

    # for k in [ 4, 5, 6, 7, 8, 9, 10]:
    for k in [3]:
        start = time.time()
        cal_similarity1 = cal_similarity("comloc")
        checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/FS_NY_1.csv", delimiter="\t", index_col=None)
        # checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/comloc/1_disturb_accuracy_3_comloc.csv", sep='\t', names=["uid", "time", "latitude", "longitude", "locid", "clusterid"], header=None)
        # cal_similarity.set_checkins(checkins.values.tolist(), "1", 30, 40, [30.387953, -97.843911, 30.249935, -97.635460])
        # cal_similarity.cal_weighted_sim(4, 0.5, 0.5)
        # checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/comloc/SNAP_NY_1_1_" + str(k) + "_comloc.csv",
        #                        sep='\t', names=["uid", "time", "latitude", "longitude", "locid", "grid_id", "clusterid","is_core", "grid_id_before", "locid_before"],
        #                        header=None)
        # checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/comloc/19_"+str(k)+"_freqloc.csv",
        #                        sep='\t', names=["uid", "time", "latitude", "longitude", "locid", "grid_id", "clusterid", "is_core", "grid_id_before", "locid_before"],
        #                        header=None)
        checkins = checkins.sort_values(by=['uid'], ascending=False).reset_index(drop=True)
        checkins = checkins.ix[:, [0, 1, 2, 3, 4]]
        # cal_similarity1.set_checkins(checkins.values.tolist(), "1", k, 30, 40, [30.387953, -97.843911, 30.249935, -97.635460])
        # cal_similarity1.set_checkins(checkins.values.tolist(), "SF", k, 30, 40, [37.809524, -122.520352, 37.708991, -122.358712])
        cal_similarity1.set_checkins(checkins.values.tolist(), "FS_NY_1", k, 40, 40, [40.836357, -74.052914, 40.656702, -73.875168])   # 20KM*
        cal_similarity1.cal_comloc_user_pairs()
        cal_similarity1.cal_freqloc_user_pairs(3)
        end = time.time()
        print(str(end-start))