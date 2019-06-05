import pandas as pd
import time
import multiprocessing
from joblib import Parallel, delayed
import itertools
filename = "file2.csv"

def user_combination(checkins):
    userlist = checkins.uid.unique()
    combinelist = list(itertools.combinations(userlist, 2))
    df = pd.DataFrame(combinelist, columns=['u1', 'u2'])
    df.to_csv(filename, sep='\t', header=True, index=False)

# def userList(checkins):
#     checkin = pd.DataFrame()
#     checkin['uid'] = checkins['uid']
#     checkin['country_id'] = checkins['country_id']
#     checkin['state_id'] = checkins['state_id']
#     checkin['index_ij'] = checkins['index_ij']
#     usersCheckinTimes_country_state = checkin.groupby(["country_id", "state_id"])
#     # 使用并行处理有问题
#     for group in usersCheckinTimes_country_state:
#         user_combination(group[1])


def meet(u1, u2, checkins):
    list1 = []
    u1_checkins = checkins[checkins.uid == u1]
    u2_checkins = checkins[checkins.uid == u2]
    u1_length = len(u1_checkins)
    u2_length = len(u2_checkins)
    meet_list = list(set(u1_checkins.index_ij.unique()).intersection(set(u2_checkins.index_ij.unique())))
    list_len = len(meet_list)
    com_indexij = ""
    u1_u2_times = ""
    for i in range(list_len):
        u1times = len(u1_checkins[u1_checkins.index_ij == str(meet_list[i])])
        u2times = len(u2_checkins[u2_checkins.index_ij == str(meet_list[i])])
        if i+1 != list_len:
            com_indexij = com_indexij + str(meet_list[i])+","
            # 访问频率的计算方法可能不是这样算的，用的是u1访问某个位置的概率*u2访问这个位置的概率
            u1_u2_times = u1_u2_times + str((u1times * u2times * 1.0)/(u1_length * u2_length)) + ","
        else:
            com_indexij = com_indexij + str(meet_list[i])
            u1_u2_times = u1_u2_times + str((u1times * u2times * 1.0) / (u1_length * u2_length))
    list1.append(str(u1))
    list1.append(str(u2))
    list1.append(len(meet_list))
    list1.append(com_indexij)
    list1.append(u1_u2_times)
    return list1



def cell_analyze(checkins):
    checkins = checkins.groupby(["uid", "index_ij"]).size().sort_values(ascending=False).reset_index(name="uid_cell_times")
    name = ['u1', 'u2']
    userlist = pd.read_csv('file2.csv', low_memory=False, delimiter="\t", header=1, names=name)
    print(len(userlist))
    print("开始分析")
    core_num = multiprocessing.cpu_count()
    meet_cell = Parallel(n_jobs=core_num)(delayed(meet)(userlist.iloc[i]['u1'], userlist.iloc[i]['u2'], checkins) for i in range(len(userlist)))
    user_indexij_result = pd.DataFrame(meet_cell)
    user_indexij_result.to_csv('result2.csv', sep='\t', index=False)
    print("一个单元格分析完成")


if __name__ == "__main__":
    start = time.time()
    columns = ['uid', 'time', 'latitude', 'longitude', 'locid', 'country_id', 'state_id', 'index_ij']
    checkins = pd.read_csv('totalCheckins.csv', low_memory=False, delimiter="\t", header=1, names=columns)
    checkin = pd.DataFrame()
    checkin['uid'] = checkins['uid']
    checkin['country_id'] = checkins['country_id']
    checkin['state_id'] = checkins['state_id']
    checkin['index_ij'] = checkins['index_ij']
    usersCheckinTimes_country_state = checkin.groupby(["country_id", "state_id"])
    # 只进行了一个州的数据处理，使用break，时间大概50分钟......
    print("开始处理")
    for group in usersCheckinTimes_country_state:
        user_combination(group[1])
        print("得到一个州内的用户")
        cell_analyze(group[1])
        break
    end = time.time()
    print("运行时间为：", str(end-start))
