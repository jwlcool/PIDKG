from cmd_args import cmd_args
from preprocess import *
RATING_FILE_NAME = dict({'movie': 'ratings.csv','movie-1m': 'ratings.csv', 'book': 'BX-Book-Ratings.csv', 'music': 'user_artists.dat'})
SEP = dict({'movie': ',' , 'movie-1m': '\t' , 'book': ';' , 'music': '\t'})
THRESHOLD = dict({'movie': 0,'movie-1m': 0, 'book': 0, 'music': 0})
DATASET = "movie-1m"

entity_id2index = dict()
relation_id2index = dict()
item_index_old2new = dict()
file = '../data/' + DATASET + '/item_index2entity_id.txt'
print('reading item index to entity id file: ' + file + ' ...')
i = 0
for line in open(file, encoding='utf-8').readlines():
    item_index = line.strip().split('\t')[0]
    satori_id = line.strip().split('\t')[1]
    item_index_old2new[item_index] = i
    entity_id2index[satori_id] = i
    i += 1

file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]
print('reading rating file ...')
item_set = set(item_index_old2new.values())
user_pos_ratings = dict()
user_neg_ratings = dict()
item_list=[]
user_list=[]
for line in open(file, encoding='utf-8').readlines()[1:]:
    array = line.strip().split(SEP[DATASET])

    # remove prefix and suffix quotation marks for BX dataset
    if DATASET == 'book':
        array = list(map(lambda x: x[1:-1], array))

    item_index_old = array[1]
    if item_index_old not in item_index_old2new:  # the item is not in the final item set
        continue
    item_index = item_index_old2new[item_index_old]

    user_index_old = int(array[0])

    rating = float(array[2])
    if rating >= THRESHOLD[DATASET]:
        if user_index_old not in user_pos_ratings:
            user_pos_ratings[user_index_old] = set()
        user_pos_ratings[user_index_old].add(item_index)
    else:
        if user_index_old not in user_neg_ratings:
            user_neg_ratings[user_index_old] = set()
        user_neg_ratings[user_index_old].add(item_index)

    item_list.append(int(item_index_old))

    user_list.append(user_index_old)
print('converting rating file ...')
user_list=list(set(user_list))
item_list = list(set(item_list))
user_list.sort()
item_list.sort()
user_cnt = 0
user_index_old2new = dict()
for user_index_old, pos_item_set in user_pos_ratings.items():
    if user_index_old not in user_index_old2new:
        user_index_old2new[user_index_old] = user_cnt
        user_cnt += 1
    user_index = user_index_old2new[user_index_old]

print('number of users: %d' % user_cnt)
print('number of items: %d' % len(item_set))

users=[]
items=[]

writer = open('../data/' + DATASET + '/ratings_new.csv', 'w', encoding='utf-8')
writer.write("user,item,rating,time")
writer.write('\n')
for line in open('../data/'+DATASET+'/ratings_order.csv', encoding='utf-8').readlines()[1:]:
    array = line.strip().split(SEP[DATASET])
    user_index_old = int(array[0])
    item_index_old = array[1]
    if item_index_old not in item_index_old2new:
        continue
    if user_index_old not in user_index_old2new:
        continue
    item_new = item_index_old2new[item_index_old]
    user_new = user_index_old2new[int(user_index_old)]
    users.append(user_new)
    items.append(int(item_new))
    st = str(user_new) + " " + str(item_new) + " " + array[2] + " " + array[3]
    writer.write(st)
    writer.write('\n')
users=list(set(users))
items=list(set(items))
users.sort()
items.sort()
writer.close()
print("0")


# user_list=[]
# item_list=[]
# rating_list=[]
# timestamp_list=[]
# new_path.write("user,item,rating,time")
# new_path.write('\n')
# with open(path, "r") as f:
#     f.readline()
#     for cnt,l in enumerate(f):
#         ls = l.strip().split(",")
#         user=user_index_old2new[ls[0]]
#         if ls[1] not in item_index_old2new:  # the item is not in the final item set
#             continue
#         item=item_index_old2new[ls[1]]
#         rating=ls[2]
#         timestamp=ls[3]
#         st=str(user)+" "+str(item)+" "+rating+" "+timestamp
#         new_path.write(st)
#         new_path.write('\n')


# path="../data/movie-1m/ratings_new.csv"
# train_path="../data/movie-1m/train.txt"
# train=open(train_path,"w")
# test_path="../data/movie-1m/test.txt"
# test=open(test_path,"w")
# count=0
#
# with open(path, "r") as f:
#     f.readline()
#     for cnt, l in enumerate(f):
#         count=count+1
# n=count*0.7
# f.close()
# print(count)
#
# with open(path, "r") as f:
#     f.readline()
#     for cnt,l in enumerate(f):
#         ls = l.strip().split(" ")
#         user=ls[0]
#         item=ls[1]
#         rating=ls[2]
#         timestamp=ls[3]
#         str=user+" "+item+" "+timestamp+" "+rating+" "+'0'+" "+'0'
#         n = n - 1
#         if n >=0:
#             train.write(str)
#             train.write('\n')
#         else:
#             test.write(str)
#             test.write('\n')

# user_list=[]
# item_list=[]
# path=open("../data/movie-1m/ratings.csv")
# new_path="../data/movie-1m/ratings_order.csv"
# result=[]
# iter_f=iter(path)
# for line in iter_f:
#     if line[0]=='u':continue
#     result.append(line)
# path.close()
# result.sort(key=lambda x:float(x.split('\t')[3]),reverse=False)
# f=open(new_path,'w')
# f.write('\n')
# f.writelines(result)
# f.close()








# path="../data/music/ratings_new.csv"
# user_list=[]
# item_list=[]
# with open(path,"r") as f:
#     f.readline()
#     for cnt, l in enumerate(f):
#         # FORMAT: user, item, timestamp, state label, feature list
#         ls = l.strip().split(' ')
#         user=ls[0]
#         item=ls[1]
#         timestamp=ls[3]
#         if int(user) not in user_list:
#             user_list.append(int(user))
#         if int(item) not in item_list:
#             item_list.append(int(item))
#     f.close()
# user_list.sort()
# item_list.sort()
# print("0")




# path=open("../data/music/user_taggedartists-timestamps.dat")
# new_path="../data/music/ratings_order.csv"
# result=[]
# iter_f=iter(path)
# for line in iter_f:
#     if line[0]=='u':continue
#     result.append(line)
# path.close()
# result.sort(key=lambda x:float(x.split('\t')[3]),reverse=False)
# f=open(new_path,'w')
# f.writelines(result)
# f.close()

#
# path="../data/music/kg.txt"
# user_list=[]
# item_list=[]
# all=[]
# same=[]
# with open(path,"r") as f:
#     f.readline()
#     for cnt, l in enumerate(f):
#         # FORMAT: user, item, timestamp, state label, feature list
#         ls = l.strip().split('\t')
#         user=ls[0]
#         item=ls[2]
#
#         if int(user) not in user_list:
#             user_list.append(int(user))
#         if int(user) not in all:
#             all.append(int(user))
#
#         if int(item) not in item_list:
#             item_list.append(int(item))
#         if int(item) not in all:
#             all.append(int(item))
#
#     f.close()
# user_list.sort()
# item_list.sort()
# all.sort()
# for i in user_list:
#     for j in item_list:
#         if i==j:
#             same.append(i)
# print("0")

# path="../data/movie-1m/ratings.dat"
# new_path=open("../data/movie-1m/ratings.csv",'w')
# for line in open(path, encoding='utf-8').readlines()[1:]:
#     array = line.strip().split('::')
#     user=array[0]
#     item=array[1]
#     rating=array[2]
#     time=array[3]
#     str=user+'\t'+item+'\t'+rating+'\t'+time
#     new_path.write(str)
#     new_path.write('\n')