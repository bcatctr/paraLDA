# process output data, find topic words
import csv
with open('./data/20news/20news.dict') as csvfile:
    reader = csv.reader(csvfile)
    vocab = []
    for words in reader:
        vocab.append(words[0])

print len(vocab)
with open('./output/result.tw') as csvfile:
    reader = csv.reader(csvfile)
    word_topic = []
    for line in reader:
        counts = []
        for count in line:
            counts.append(int(count))
        word_topic.append(counts)

# process output data, find topic words

topic_word = zip(*word_topic)
for (k,count) in enumerate(topic_word):
    max_3 = sorted(enumerate(count), key = lambda (i,c):c, reverse = True)
    print "###### Topic",k,"######"
    for item in max_3[:5]:
        print vocab[item[0]],item[1]