#!/usr/bin/env python
# encoding: utf-8

import sqlite3 as db
import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy
import multiprocessing
from joblib import Parallel,delayed
from sys import argv


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


class DP_security():

    def __init__(self, tableName, tableName_obf):
        dbName = "G:/pyfile/relation_protect/src/data/city_data/" + tableName_obf + ".db"  # 原始数据
        conn = db.connect(dbName)
        sql_select = ' '.join(['SELECT * FROM', tableName_obf, ';'])
        self.checkins_obf = pd.read_sql_query(sql_select, conn)
        self.checkins_obf.columns = ["uid", "locid", "time", "longitude", "latitude", "semantic"]
        sql_select = ' '.join(['SELECT * FROM', tableName, ';'])
        self.checkins = pd.read_sql_query(sql_select, conn)
        self.checkins.columns = ["userid", "locid", "placeid", "dtime", "lon_ori", "lat_ori", "lon_fal", "lat_fal","semantic_ori", "semantic_fal", "dist"]
        self.city = tableName_obf
        conn.close()

    def a_security(self):
        print()
        print("用户间共同访问位置改变情况")
        similarity_pairs = pd.read_csv(
            "G:/pyfile/relation_protect/src/data/city_data/" + self.city + "_comloc.similarity",
            names=["u1", "u2", "similarity"])
        pairs = similarity_pairs[similarity_pairs.similarity > 0]
        pairs = pairs.ix[:, [0, 1]]
        core_num = multiprocessing.cpu_count()
        meet_cell = Parallel(n_jobs=core_num)(delayed(single_security)(
            self.checkins.loc[self.checkins.userid == pairs.iloc[i]['u1']].placeid.unique(),
            self.checkins.loc[self.checkins.userid == pairs.iloc[i]['u2']].placeid.unique(),
            self.checkins_obf.loc[self.checkins_obf.uid == pairs.iloc[i]['u1']].locid.unique(),
            self.checkins_obf.loc[self.checkins_obf.uid == pairs.iloc[i]['u2']].locid.unique(), i
        ) for i in range(len(pairs)))
        security = sum(meet_cell) / len(pairs)
        print("a:", security)
        return security

    def b_security(self, m):
        single_security = 0
        for user in self.checkins_obf.uid.unique():
            u_checkin = self.checkins[self.checkins.userid == user]
            u_checkin_obf = self.checkins_obf[self.checkins_obf.uid == user]
            u_loc_distr = pd.DataFrame(u_checkin['placeid'].value_counts()).reset_index()  # 统计locid的不同值及其个数
            u_loc_distr.columns = ['placeid', 'cnt']
            u_loc_distr = u_loc_distr.sort_values(by=['cnt', 'placeid'], ascending=False).reset_index(drop=True)
            u_loc_distr_obf = pd.DataFrame(u_checkin_obf['locid'].value_counts()).reset_index()
            u_loc_distr_obf.columns = ['locid', 'cnt']
            u_loc_distr_obf = u_loc_distr_obf.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
            if m <= len(u_loc_distr):
                u_loc_distr = u_loc_distr[0:m]
            if m <= len(u_loc_distr_obf):
                u_loc_distr_obf = u_loc_distr_obf[0:m]
            itstlist = list(set(u_loc_distr['placeid'].values).intersection(set(u_loc_distr_obf['locid'].values)))
            single_security += len(itstlist) * 1.0/len(u_loc_distr_obf)
        security = single_security / len(self.checkins_obf.uid.unique())
        print("b:", security)
        return security


    def c_security(self, m):
        print("全局的频繁访问位置分布改变情况")
        checkin = self.checkins
        checkin_obf = self.checkins_obf
        checkin_fre = pd.DataFrame(checkin['placeid'].value_counts()).reset_index()
        checkin_fre.columns = ['placeid', 'cnt']
        checkin_fre = checkin_fre.sort_values(by=['cnt', 'placeid'], ascending=False).reset_index(drop=True)
        checkin_obf_fre = pd.DataFrame(checkin_obf['locid'].value_counts()).reset_index()
        checkin_obf_fre.columns = ['locid', 'cnt']
        checkin_obf_fre = checkin_obf_fre.sort_values(by=['cnt', 'locid'], ascending=False).reset_index(drop=True)
        checkin_fre = checkin_fre[0:m]
        checkin_obf_fre = checkin_obf_fre[0:m]
        itstlist = list(set(checkin_fre['placeid'].values).intersection(set(checkin_obf_fre['locid'].values)))
        globle_security = len(itstlist) * 1.0 / len(checkin_obf_fre)
        print("c:", globle_security)
        return globle_security
        pass

    def d_security(self):
        print()
        print("全局位置访问频率分布改变情况")   # 全局位置访问频率分布改变情况，保护前后访问频率分布分别为Ta,Tb
        checkin = self.checkins
        checkin_obf = self.checkins_obf
        checkin_len = len(checkin)
        checkin_obf_len = len(checkin_obf)
        union_grid_id = list(set(checkin.placeid.unique()).union(set(checkin_obf.locid.unique())))
        checkin_vec = list(map((lambda x: len(checkin[checkin.placeid == x]) * 1.0 / checkin_len), union_grid_id))
        checkin_obf_vec = list(map((lambda x: len(checkin_obf[checkin_obf.locid == x]) * 1.0 / checkin_obf_len), union_grid_id))
        checkin_vec = np.array(list(checkin_vec))
        checkin_obf_vec = np.array(list(checkin_obf_vec))
        globle_security = JSD(checkin_vec, checkin_obf_vec)
        # end = time.time()
        # print("总共花的费时间为", str(end - start))
        print("d:", globle_security)
        return globle_security

    def e_security(self):
        print()
        print("用户位置访问频率分布改变情况")
        checkin = self.checkins
        checkin_obf = self.checkins_obf
        single_security = 0
        for u in checkin_obf.uid.unique():
            u_checkin = checkin.loc[checkin.userid == u]
            u_checkin_obf = checkin_obf.loc[checkin_obf.uid == u]
            u_checkin_len = len(u_checkin)
            u_checkin_obf_len = len(u_checkin_obf)
            union_grid_id = list(set(u_checkin.placeid.unique()).union(set(u_checkin_obf.locid.unique())))
            checkin_vec = list(map((lambda x: len(u_checkin[u_checkin.placeid == x]) * 1.0 / u_checkin_len), union_grid_id))
            checkin_obf_vec = list(map((lambda x: len(u_checkin_obf[u_checkin_obf.locid == x]) * 1.0 / u_checkin_obf_len), union_grid_id))
            single_security += JSD(np.array(checkin_vec), np.array(checkin_obf_vec))
        print("e:", single_security/len(checkin_obf.uid.unique()))
        return single_security/len(checkin_obf.uid.unique())

    def g_secutity(self):
        distance = self.checkins.dist.sum()/1000
        print("dis", distance)
        return distance

    def security_unility(self):
        a = self.a_security()
        b = self.b_security(3)
        c = self.c_security(100)
        d = self.d_security()
        e = self.e_security()
        g = self.g_secutity()
        file = open("G:/pyfile/relation_protect/src/data/result_data/DP.txt", 'a', encoding='UTF-8')
        file.write('安全性:' + ' ' + 'a:' + str(a) + ' ' + 'b:' + str(b) + ' ' + 'c:' + str(c)
                   + ' ' + 'd:' + str(d) + ' ' + 'e:' + str(e)
                   + ' ' + 'g:' + str(g)
                   + '\n')
        file.close()


if __name__ == "__main__":
    # cluster_num = [31, 17, 27, 33]
    j = 0
    # i=argv[1]
    cluster_num = [15]
    city="Gowalla_AS"
    for city in ["Gowalla_AS", "SF", "FS_NY_1"]:
    # for city in ["Gowalla_AS"]:
    #     print(city)
        for i in ["01", "1", "3", "5", "10"]:
                dp_se = DP_security("dp_result_" + str(cluster_num[j])+"_300_" + city + i, city)
                dp_se.security_unility()
    # dp_se.b_security(3)
    # dp_se.c_security(100)
    # dp_se.g_secutity()
    # dp_se.d_security()
    # dp_se.e_security()
    # dp_se.a_security()
        # j += 1