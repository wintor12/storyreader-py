import os
import codecs
from datetime import date


month2num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
week2num = {'Fri': 19, 'Sat': 20, 'Sun': 21, 'Mon': 22, 'Tue': 23, 'Wed': 24,
            'Thu': 25}
current_date = date(2016, 8, 26)


path = './all_data'
src_path = './all_data/s_train'
question_path = './all_data/q_train'
feature_path = './all_data/feature_train'
tgt_path = './all_data/y_train'
time_path = './all_data/time_train.txt'
title_path = './all_data/title_train.txt'


def loadTime(time_path):
    times = []
    with codecs.open(time_path, 'r', 'utf-8') as p:
        for time in p:
            times.append(time.strip())
    return times


def loadData(src_path, question_path, feature_path, tgt_path, time_path):
    examples = []
    with codecs.open(src_path, 'r', 'utf-8') as src_file, \
         codecs.open(question_path, 'r', 'utf-8') as q_file, \
         codecs.open(feature_path, 'r', 'utf-8') as f_file, \
         codecs.open(tgt_path, 'r', 'utf-8') as t_file, \
         codecs.open(time_path, 'r', 'utf-8') as ti_file, \
         codecs.open(title_path, 'r', 'utf-8') as title_file:
        for i, (src_line, q_line, f_line, t_line, ti_line, title_line) in enumerate(
                zip(src_file, q_file, f_file, t_file, ti_file, title_file)):
            src = src_line.strip().split()
            question = q_line.strip().split()
            feature = f_line.strip().split()
            feature = [float(x) for x in feature]
            tgt = float(t_line.strip())
            time = ti_line.strip()
            title = title_line.strip()
            d = {'src': src, 'question': question, 'indices': i,
                 'view': feature[0], 'pics': feature[1],
                 'tgt': tgt, 'time': time, 'title': title}
            examples.append(d)
    return examples


def preprocessTime(time, current_date):
    time = time[8:]  # remove written/updated
    month = time[:3]
    if month in week2num:  # week format, the last week
        year = 2016
        day = week2num[month]
        month = 8
    elif month in month2num:
        month = int(month2num[month])
        time = time.split(' ')[1:]
        if len(time) == 1:
            year = 2016
            day = int(time[0])
        else:
            year = int(time[1])
            day = int(time[0][:-1])
    else:
        month, day, year = 8, 26, 2016
    d = date(year, month, day)
    if (current_date - d).days < 0:
        year -= 1
    return date(year, month, day)
        

def main():
    # train_examples = loadData(src_path, question_path,
    # feature_path, tgt_path, time_path)
    # print(train_examples[12000])
    # times = [x['time'] for x in train_examples]
    times = loadTime(time_path)
    print(len(times))
    times = [preprocessTime(time, current_date) for time in times]
    times_diff = [(current_date - time).days for time in times]
    print(times[times_diff.index(2350)])
    print(min(times_diff))
    print(times[times_diff.index(min(times_diff))])


if __name__ == '__main__':
    main()
