import datetime


def scaling_int(int_num, scale):
    return int(int_num * scale)


def unix_time_to_milliseconds(dt, epoch):
    return (dt - epoch).total_seconds() * 1000.0