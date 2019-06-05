#!/usr/bin/env python
# encoding: utf-8

from math import cos, sin, atan2, sqrt, pi, radians, degrees
from joblib import Parallel, delayed
import multiprocessing as mp
import pandas as pd

class get_center_point():

    def __init__(self, checkins):
        self.checkins_obf = checkins
        pass

    def center_geolocation(self, geolocations, u):
        x = 0
        y = 0
        z = 0
        lenth = len(geolocations)
        for lat, lon in geolocations:
            lon = radians(float(lon))
            lat = radians(float(lat))
            x += cos(lat) * cos(lon)
            y += cos(lat) * sin(lon)
            z += sin(lat)
        x = float(x / lenth)
        y = float(y / lenth)
        z = float(z / lenth)
        return [u, degrees(atan2(z, sqrt(x * x + y * y))), degrees(atan2(y, x))]

    def user_core_point(self, users):
        # users = list(self.checkins_obf.uid.unique())
        # core_nums = mp.cpu_count()
        # u_center_point = Parallel(n_jobs=core_nums)(delayed(self.center_geolocation)(self.checkins_obf[self.checkins_obf.uid == u].groupby(by=['locid', 'latitude', 'longitude']).size().reset_index(name='freq'), u) for u in users)
        u_center_point = []
        for u in users:
            u_checkins = self.checkins_obf[self.checkins_obf.uid == u]
            u_locid = u_checkins.groupby(by=['locid', 'latitude', 'longitude']).size().reset_index(name='freq')
            u_locations = [[row[1], row[2]] for row in u_locid.itertuples(index=False, name=False)]
            u_center_point.append(self.center_geolocation(u_locations, u))
        u_center_point = pd.DataFrame(u_center_point, columns=['uid', 'latitude', 'longitude'])
        return u_center_point