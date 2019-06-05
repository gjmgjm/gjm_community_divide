#!/usr/bin/env python
# encoding: utf-8


import pandas as pd
from joblib import Parallel,delayed
import multiprocessing as mp
from itertools import combinations
import math


class protect_freqloc:

    def __init__(self):
        pass


    def update_pairs(self):
        pass


    def k_freq_loc(self, u, u_checkins, k):
        u_loc = pd.DataFrame(u_checkins['locid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
        u_loc.columns = ["locid", "cnt"]
        if k <= len(u_loc):
            u_loc = u_loc[0:k]
        return [u, list(u_loc.locid.values)]

    def u_comloc(self, u1_checkins, u2_checkins, u1, u2):
        u1_locids = u1_checkins.locid.unique()
        u2_locids = u2_checkins.locid.unique()
        comloclen = list(set(u1_locids).intersection(set(u2_locids)))
        return [u1, u2, len(comloclen), comloclen]

    def freqloc_protection(self, checkins, k, m):  # k为用户的前k个频繁访问位置、m为前m%的共同访问位置，checkins是社区内数据
        uids = checkins.uid.unique() # 用户id进行排序
        core_num = mp.cpu_count()
        user_k_freqloc = Parallel(n_jobs=core_num)(delayed(self.k_freq_loc)(checkins[checkins.uid == u], u, 2) for u in uids)
        user_k_freqloc = pd.DataFrame(user_k_freqloc, columns=['uid', 'k_freqlocs'])
        pairs = list(combinations(uids, 2))
        pairs = pd.DataFrame(pairs, columns=['u1', 'u2'])
        pairs_comloc = Parallel(n_jobs=core_num)(delayed(self.u_comloc)(checkins[checkins.uid == row[0]], checkins[checkins.uid == row[1]], row[0], row[1]) for row in pairs.itertuples(index=False, name=False))
        pairs_comloc = pd.DataFrame(pairs_comloc, columns=["u1", "u2", "comloc_num","comlocs"])
        nums = int(math.ceil(len(pairs_comloc) * 0.2))
        pairs_comloc = pairs_comloc[0:nums].reset_index()  # 20%的共同访问位置用户对,即使需要扰动的共同访问位置的用户对
        for row in pairs_comloc.itertuples(index=False,name=False):
            pass