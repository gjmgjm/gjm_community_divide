#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
from grid_divide import grid_divide
import time
from sys import argv

class random_disturb_security():

    def __init__(self, checkins, city, m, n, range, lats_per_km, lons_per_km):
        self.checkin_obf = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/" + city + ".csv", delimiter="\t",index_col=None)
        # self.m = 40
        # self.n = 30
        # self.range = [30.387953, -97.843911, 30.249935, -97.635460]
        self.checkins = checkins
        self.m = m
        self.n = n
        self.range = range
        self.lons_per_km = lons_per_km
        self.lats_per_km = lats_per_km
        self.city = city
        # self.checkin_obf = self.checkin_obf.values.tolist()
        # self.checkin_obf = grid_divide(self.checkin_obf, 30, 40, [30.387953, -97.843911, 30.249935, -97.635460]).divide_area_by_NN()

    def cal_security(self, k):

        print(k)
        from security_unility import security_unility
        security = security_unility(self.lons_per_km, self.lats_per_km)
        # checkin = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/random_disturb/" + self.city + "_" + str(k) + "_random_disturb.checkin", delimiter="\t", names=['uid', 'time', 'latitude', 'longitude', 'locid', 'num'], header=None)
        checkin = self.checkins.ix[:, [0, 1, 2, 3, 4]]
        security.set_checkins(checkin, self.checkin_obf, "G:/pyfile/relation_protect/src/data/city_data/", self.city)
        security.set_grid(self.m, self.n, self.range)
        b = security.b_security(3)
        c = security.c_security(100)
        d = security.d_security()
        e = security.e_security()
        f = security.f_security2()
        g = security.g_security2()
        a = security.unility()
        # security.all_security(0, b, c, d, e, 1/3, 1/3, 1/3)
        file = open("G:/pyfile/relation_protect/src/data/result_data/random_disturb_result.txt",  'a', encoding='UTF-8')
        file.write('扰动比例参数:' + str(k) + ' ' + 'a:' + str(a) + ' ' + 'b:' + str(b) + ' ' + 'c:' + str(c)
                   + ' ' + 'd:' + str(d) + ' ' + 'e:' + str(e)
                   + ' ' + 'f:' + str(f) + ' ' + 'g:' + str(g)
                   + '\n')
        # file.write('c:' + str(k) + ' ' + str(c) + '\n')
        # file.write('a:' + str(k) + ' ' + str(a) + '\n')
        # file.write('g:' + str(k) + ' ' + str(g) + '\n')
        file.close()

        # security.all_security(a, b, c, d, e, 1/3, 1/3, 1/3)

    def cal_utility(self, k):
        print()
        print(k)
        from utility import utility
        utility = utility()
        checkin = pd.read_csv("G:/pyfile/relation_protect/src/data/result_data/random_disturb/1_" + str(k) + "_random_disturb.checkin",
                              delimiter="\t", names=['uid', 'time', 'latitude', 'longitude', 'locid', 'num'],
                              header=None)
        checkin.drop("num", axis=1, inplace=True)
        utility.set_checkins(checkin, self.checkin_obf, "G:/pyfile/relation_protect/src/data/city_data/", "G:/pyfile/relation_protect/src/data/result_data/", "1", str(k) + "_random_", "random_disturb")
        # utility.checkin_sim_list2()
        k_utility = utility.sim_utility2()
        file = open("G:/pyfile/relation_protect/src/data/result_data/random_disturb_result.txt",  'a', encoding='UTF-8')
        file.write('扰动比例参数:' + str(k) + ' ' + str(k_utility) + '\n')
        file.close()


if __name__ == "__main__":
    start = time.time()
    # k=argv[1]
    for k in [10, 20, 30, 40, 50, 60, 70, 80]:
        # random_disturb = random_disturb_security("FS_NY_1", 40, 40, [40.836357, -74.052914, 40.656702, -73.875168], 0.0044966 * 2, 0.0059352 * 2)
        # random_disturb = random_disturb_security("SNAP_NY_1", 40, 40, [40.836357, -74.052914, 40.656702, -73.875168], 0.0044966 * 2, 0.0059352 * 2)
        # random_disturb = random_disturb_security("SF", 40, 30, [37.809524, -122.520352, 37.708991, -122.358712], 0.004492 * 2, 0.005681 * 2)
        random_disturb = random_disturb_security("1", 40, 30, [30.387953, -97.843911, 30.249935, -97.635460], 0.004492 * 2, 0.005202 * 2)
        random_disturb.cal_security(k)
        # random_disturb.cal_utility(k)
        end = time.time()
    # print("总共花的费时间为", str(end-start))