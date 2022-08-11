small_userprofile='./data/movie-1m/users.dat'
DATASET='movie-1m'
THRESHOLD = dict({'movie': 0,'movie-1m': 0, 'book': 0, 'music': 0})
entity_id2index = dict()
relation_id2index = dict()
item_index_old2new = dict()
file = './data/' + DATASET + '/item_index2entity_id.txt'
print('reading item index to entity id file: ' + file + ' ...')
i = 0
for line in open(file, encoding='utf-8').readlines():
    item_index = line.strip().split('\t')[0]
    satori_id = line.strip().split('\t')[1]
    item_index_old2new[item_index] = i
    entity_id2index[satori_id] = i
    i += 1

file = './data/' + DATASET + '/' + 'ratings.csv'

print('reading rating file ...')
item_set = set(item_index_old2new.values())
user_pos_ratings = dict()
user_neg_ratings = dict()
item_list=[]
user_list=[]
for line in open(file, encoding='utf-8').readlines()[1:]:
    array = line.strip().split('\t')


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

    item_list.append(int(item_index))
    user_list.append(user_index_old)
print('converting rating file ...')
user_list = list(set(user_list))
item_list = list(set(item_list))
user_list.sort()
item_list.sort()
writer = open('./data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
user_cnt = 0
user_index_old2new = dict()
for user_index_old, pos_item_set in user_pos_ratings.items():
    if user_index_old not in user_index_old2new:
        user_index_old2new[user_index_old] = user_cnt
        user_cnt += 1
    user_index = user_index_old2new[user_index_old]

    for item in pos_item_set:
        writer.write('%d\t%d\t1\n' % (user_index, item))
    unwatched_set = item_set - pos_item_set
    if user_index_old in user_neg_ratings:
        unwatched_set -= user_neg_ratings[user_index_old]

writer.close()

print('number of users: %d' % user_cnt)
print('number of items: %d' % len(item_set))

entity_cnt = len(entity_id2index)
relation_cnt = 0
relation_dict=dict()

writer = open('./data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')
for line in open('./data/' + DATASET + '/kg.txt', encoding='utf-8'):
    array = line.strip().split('\t')
    head_old = array[0]
    relation_old = array[1]
    tail_old = array[2]


    if head_old not in entity_id2index:#知识图谱节点id
        entity_id2index[head_old] = entity_cnt
        entity_cnt += 1
    head = entity_id2index[head_old]

    if tail_old not in entity_id2index:
        entity_id2index[tail_old] = entity_cnt
        entity_cnt += 1
    tail = entity_id2index[tail_old]


    if relation_old not in relation_id2index:
        relation_id2index[relation_old] = relation_cnt
        relation_cnt += 1
    relation = relation_id2index[relation_old]

    writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    relation_dict[(head,tail)]=relation

writer.close()


print('number of entities (containing items): %d' % entity_cnt)
print('number of relations: %d' % relation_cnt)


user_kg_gender2entity_id=dict()
user_kg_age2entity_id=dict()
user_kg_occupation2entity_id=dict()
user_kg_pvalue_level2entity_id = dict()
user_kg_shopping_level2entity_id = dict()
user_kg_user_class_level2entity_id = dict()
user_kg=dict()
user_kg_writer=open('./data/' + DATASET + '/kg_user.txt', 'w', encoding='utf-8')
user_file = './data/' + DATASET + '/' + 'users.dat'
for line in open(user_file, encoding='utf-8').readlines():
    array = line.strip().split('::')
    user = int(array[0])
    gender = array[1]
    age = int(array[2])
    occupation = int(array[3])
    user_id=user_index_old2new[user]
    if gender not in user_kg_gender2entity_id:
        # entity_id2index[gender] = entity_cnt
        user_kg_gender2entity_id[gender]=entity_cnt
        entity_cnt += 1
    gender_id = user_kg_gender2entity_id[gender]
    gender_id_relation = str(user) + "\t" + "user.gender" + "\t" + str(gender_id)
    user_kg_writer.write(gender_id_relation)
    user_kg_writer.write('\n')

    if age not in user_kg_age2entity_id:
        # entity_id2index[age] = entity_cnt
        user_kg_age2entity_id[age] = entity_cnt
        entity_cnt += 1
    age_id = user_kg_age2entity_id[age]
    age_id_relation = str(user) + "\t" + "user.age" + "\t" + str(age_id)
    user_kg_writer.write(age_id_relation)
    user_kg_writer.write('\n')

    if occupation not in user_kg_occupation2entity_id:
        # entity_id2index[occupation] = entity_cnt
        user_kg_occupation2entity_id[occupation] = entity_cnt
        entity_cnt += 1
    occupation_id = user_kg_occupation2entity_id[occupation]
    occupation_id_relation = str(user) + "\t" + "user.occupation" + "\t" + str(occupation_id)
    user_kg_writer.write(occupation_id_relation)
    user_kg_writer.write('\n')

    if user_id in user_kg: continue
    else:
        user_kg[user_id]=[gender_id,age_id,occupation_id]





