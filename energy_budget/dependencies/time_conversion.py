import numpy as np
import datetime as dt

def convert_time_to_date(time):
     date = [dt.datetime.fromordinal(int(time0)) + dt.timedelta(time0%1) for time0 in  time] 
     return date
def convert_date_to_time(date):
     N = len(date)
     time = np.full(N, np.nan)
     for i in range(N):
          time[i]=date[i].toordinal() + date[i].hour/24. + date[i].minute/24./60. + date[i].second/24./60./60. + date[i].microsecond/24./60./60./1e6
     return time

 
