#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@author: jaculamata 
@license: Apache Licence  
@contact: 819436557@qq.com 
@site: http://blog.csdn.net/hqzxsc2006 
@software: PyCharm 
@file: 1_data_process.py 
@time: 2019/1/20 15:38 
"""
from  itertools import combinations
import pandas as pd



class data_process():
    def __init__(self):
        self.checkins = None
        self.path = None
        self.city = None
        self.pairs_path = "G:/pyfile/relation_protect/src/data/city_data/"
        pass

    def set_basic_info(self, path, city, pairs_path=None):
        self.checkins = pd.read_csv(path + "city_data/" + city + ".csv", delimiter="\t", index_col=None)
        self.path = path
        self.city = city

    def set_pairs_path(self, path):
        self.pairs_path = path

    def get_checkins(self):
        return self.checkins

    def user_pairs(self):
        users = self.checkins.uid.unique()
        pairs = list(combinations(users, 2))
        pd.DataFrame(pairs).to_csv(self.pairs_path + self.city + ".pairs", index=False,
                                   header=False)
        pairs_path = "G:/pyfile/relation_protect/src/data/city_data/" + self.city + ".pairs"
        return pairs_path

