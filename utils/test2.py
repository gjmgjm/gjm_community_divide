#!/usr/bin/env python
# encoding: utf-8


import time
import math
from community_divide4 import community_divide2


class test1():

    def __init__(self):
        import pandas as pd
        # import  numpy as np
        # data = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/FS_NY_1.csv", index_col=False,sep='\t')
        # randata=data[3000:4000]
        # # randata=data.sample(2000)
        # randata.to_csv("G:/pyfile/relation_protect/src/data/city_data/1.2.csv",sep='\t',index=False)
        pass


if __name__ == "__main__":
    start = time.time()
    deta = math.exp(-1 / 3)
    t = math.exp(-0.8)
    # for l in [3, 4, 5, 6, 7, 8, 9]:
    for l in [9, 8, 7]:
        for i in [81, 82, 83, 84, 85]:
        # for i in [41]:
        #for a in [0.1, 0.2, 0.3, 0.4, 0.7, 0.8]:
        # for a in [0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54]:
        # for a in [0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.601, 0.602, 0.603, 0.604, 0.605, 0.609, 0.61, 0.62, 0.7, 0.75, 0.8]:
            start1 = time.time()
            test = community_divide2()
            # 奥斯汀
            # test.set_checkins("G:/pyfile/relation_protect/src/data/", "1", 30, 40, [30.387953, -97.843911, 30.249935, -97.635460])
            # test.community_divide_core(math.exp(-1/2), l, math.exp(-1/2), 0.5, 0.5, "comloc", 3, i) #500
            # test.community_divide_core(0.6, i, a, 0.5, 0.5, "comloc", 3)  # 500
            # test.community_divide_core(math.exp(-1/3), i, math.exp(-0.4), 0.5, 0.5, "comloc", 3)  # 400米
            # test.community_divide_core(math.exp(-0.5), i, math.exp(-1/3), 0.5, 0.5, "freqloc", 3)   # m,q
            # 旧金山
            # test.set_checkins("G:/pyfile/relation_protect/src/data/", "SF", 30, 40, [37.809524, -122.520352, 37.708991, -122.358712])
            # test.community_divide_core(math.exp(-1/4), l, math.exp(-1/2), 0.5, 0.5, "comloc", 3, i)  # 500
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
            test.set_checkins("G:/pyfile/relation_protect/src/data/", "FS_NY_1", 40, 40, [40.836357, -74.052914, 40.656702, -73.875168])
            test.community_divide_core(deta, l, t, 0.5, 0.5, "comloc", 3, i)  # 500
            # test.community_divide_core(math.exp(-1/3), i, math.exp(-0.8), 0.5, 0.5, "comloc", 3)  # 500
            # test.community_divide_core(math.exp(-1/3), i, math.exp(-0.4), 0.5, 0.5, "comloc", 3)  # 400米
            # test.community_divide_core(math.exp(-0.8), i, math.exp(-1/3), 0.5, 0.5, "freqloc", 3)  # m,q
            print(str(time.time()-start1))
    end = time.time()
    print(str(end-start))