#!/usr/bin/env python
# encoding: utf-8

import sqlite3 as db
import pandas as pd
import numpy as np


class data_read():

    def __init__(self, city):
        self.city = city
        pass

    def read_db(self):

        checkins = pd.read_csv("G:/pyfile/relation_protect/src/data/city_data/1.csv", index_col=False, sep='\t')
        checkins['semantic'] = "none"
        temp_checkins = checkins.ix[:, [0, 4, 1, 3, 2, 5]]
        temp_checkins = temp_checkins[0:5000]
        db_name = "G:/pyfile/relation_protect/src/data/city_data/" + self.city + ".db"
        conn = db.connect(db_name)
        cursor = conn.cursor()
        sql_create = ' '.join(['DROP TABLE IF EXISTS ', self.city, ';CREATE TABLE ', self.city, '(uid int, locid text, time TEXT, longitude real, latitude real, semantic TEXT);'])
        print(sql_create)
        cursor.executescript(sql_create)
        temp_checkins.to_sql(name=self.city, con=conn, if_exists='append', index=False)
        # cursor.executescript(sql_create)
        # data = pd.read_sql_query("select * from Checkins;", conn)
        # print(len(data))
        # data.columns = ["uid", "time", "latitude", "longitude", "locid"]
        # print(len(data.locid.unique()))
        # print(len(data.uid.unique()))
        # user_locids = data.groupby(by=["uid", "locid"]).size().reset_index(name="times")
        # temp_user = user_locids.groupby(by=['uid']).size().reset_index(name="times")
        # temp_user = temp_user[temp_user.times >= 3]
        # uids = list(temp_user.uid.unique())
        # user_locids = user_locids[user_locids.uid.isin(uids)]
        # print(np.min(user_locids.times))
        # locid_nums = data.groupby("locid").size().reset_index(name="locid_times")
        # print("位置最小签到数", np.min(locid_nums.locid_times))
        # users = list(data.uid.unique())
        # all_data = pd.read_sql_query("select * from Checkins;", conn)
        # all_data.columns = ["uid", "time", "latitude", "longitude", "locid"]
        # choose_datas = pd.DataFrame()
        # for row in user_locids.itertuples(index=False, name=False):
        #     choose_data = all_data[(all_data.uid == row[0]) & (all_data.locid == row[1])]
        #     choose_datas = pd.concat([choose_datas, choose_data], ignore_index=True)
        # choose_datas.reset_index(drop=True)
        # checkins = all_data[all_data.uid.isin(users)]
        # choose_datas.to_csv("G:/pyfile/relation_protect/src/data/city_data/SF.csv", header=True, index=False, sep="\t")
        # data.to_csv("G:/pyfile/relation_protect/src/data/origin_data/densely_Gowalla.csv", header=True, index=False, sep="\t")
        # conn.close()


if __name__ == "__main__":

    data_read = data_read("_5BBB")
    data_read.read_db()