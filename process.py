import codecs


# load dictionary
voc = {}
with codecs.open('data/dictionary', 'r', 'utf-8') as dic:
    for line in dic:
        word, index = line.strip().split(' ')
        voc[index] = word


def processText(src, tgt):
    res = []
    with codecs.open('data/' + src, 'r', 'utf-8') as p:
        for line in p:
            res.append(' '.join([voc[x] for x in line.strip().split(' ')]))
    with codecs.open('data/' + tgt, 'w', 'utf-8') as p:
        p.write('\n'.join(res))


processText('question_train', 'q_train')
processText('question_val', 'q_val')
processText('question_test', 'q_test')
processText('story_train', 's_train')
processText('story_val', 's_val')
processText('story_test', 's_test')
