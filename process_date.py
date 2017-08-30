import codecs
from datetime import date


month2num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
week2num = {'Fri': 19, 'Sat': 20, 'Sun': 21, 'Mon': 22, 'Tue': 23, 'Wed': 24,
            'Thu': 25}
current_date = date(2016, 7, 1)
current_date2 = date(2017, 4, 1)


path = './all_data'
src_path = './all_data/s_train'
question_path = './all_data/q_train'
feature_path = './all_data/feature_train'
tgt_path = './all_data/y_train'
time_train_path = './all_data/time_train.txt'
time_val_path = './all_data/time_val.txt'
time_test_path = './all_data/time_test.txt'
time2_train_path = './all_data/time_train.txt'
time2_val_path = './all_data/time_train.txt'
time2_test_path = './all_data/time_train.txt'
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


def saveTime(time_path, save_path, current_time):
    times = loadTime(time_path)
    print(len(times))
    times = [str(preprocessTime(time, current_date)) for time in times]
    with open(save_path, 'w') as p:
        p.write('\n'.join(times))


def saveTime2(title_path, date_path, title2_path, save_path):
    '''
    find corresponding date from the titles in
    old version dataset and copy to new version dataset
    '''
    with open(title_path) as p:
        titles = p.readlines()
    with open(date_path) as p:
        dates = p.readlines()
    with open(title2_path) as p:
        titles2 = p.readlines()
    print(len(titles), len(dates), len(titles2))
    dates2 = [dates[titles.index(t)].strip() for t in titles2]
    print(len(dates2))
    with open(save_path, 'w') as p:
        p.write('\n'.join(dates2))


def processDate(date_path, save_path, current_date):
    with open(date_path) as p:
        dates = p.readlines()
    dates = [d.strip().split('-') for d in dates]
    days = [str((current_date - date(int(d[0]), int(d[1]), int(d[2]))).days)
            for d in dates]
    with open(save_path, 'w') as p:
        p.write('\n'.join(days))


def main():
    # train_examples = loadData(src_path, question_path,
    # feature_path, tgt_path, time_path)
    # print(train_examples[12000])
    # times = [x['time'] for x in train_examples]
    # saveTime(time_train_path, './all_data/date_train', current_date)
    # saveTime(time_val_path, './all_data/date_val', current_date)
    # saveTime(time_test_path, './all_data/date_test', current_date)

    # times_diff = [(current_date - time).days for time in times]

    # saveTime2('./all_data/title_train.txt', './all_data/date_train',
    #           './all_data/title2_train.txt', './all_data/date2_train')

    # saveTime2('./all_data/title_val.txt', './all_data/date_val',
    #           './all_data/title2_val.txt', './all_data/date2_val')

    # saveTime2('./all_data/title_test.txt', './all_data/date_test',
    #           './all_data/title2_test.txt', './all_data/date2_test')

    processDate('./all_data/date_train', './all_data/day_train', current_date)
    processDate('./all_data/date_val', './all_data/day_val', current_date)
    processDate('./all_data/date_test', './all_data/day_test', current_date)

    processDate('./all_data/date2_train', './all_data/day2_train', current_date2)
    processDate('./all_data/date2_val', './all_data/day2_val', current_date2)
    processDate('./all_data/date2_test', './all_data/day2_test', current_date2)


if __name__ == '__main__':
    main()
