from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import networkx as nx
import sys
import gc
import torch
import numpy as np
import random

from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# from data_loader import load_kg


current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + "../")
sys.path.append(root_path)

from coevolve.common.cmd_args import cmd_args
from coevolve.common.consts import DEVICE
from coevolve.common.dataset import *
from coevolve.common.bipartite_graph import bg
from coevolve.common.recorder import cur_time, dur_dist
from coevolve.model.deepcoevolve import DeepCoevolve
from tqdm import tqdm

RATING_FILE_NAME = dict({'movie': 'ratings.csv','movie-1m': 'ratings.csv', 'book': 'BX-Book-Ratings.csv', 'music': 'user_artists.dat',
                        'Taobao': 'rating.csv','Taobao_Big': 'rating.csv'
                         })
SEP = dict({'movie': ',' , 'movie-1m': '\t' , 'book': ';' , 'music': '\t', 'Taobao': '\t','Taobao_Big': '\t'})
THRESHOLD = dict({'movie': 0,'movie-1m': 0, 'book': 0, 'music': 0})
DATASET = cmd_args.dataset

def load_data():




    train_data.load_events(1,cmd_args.train_file, 'train')
    test_data.load_events(1,cmd_args.test_file, 'test')

    for e_idx, cur_event in enumerate(test_data.ordered_events):
        cur_event.global_idx += train_data.num_events
        if cur_event.prev_user_event is None:
            continue
        train_events = train_data.user_event_lists[cur_event.user]
        if len(train_events) == 0:
            continue
        assert train_events[-1].t <= cur_event.t
        cur_event.prev_user_event = train_events[-1]
        cur_event.prev_user_event.next_user_event = cur_event

    print('# train:', train_data.num_events, '# test:', test_data.num_events)
    print('totally', cmd_args.num_users, 'users,', cmd_args.num_items, 'items')


def main_loop():
    sum_HR_5=0
    result=cmd_args.save_dir+'result_%s.txt'%(cmd_args.dataset)
    f = open(result, 'w')
    writer = SummaryWriter('D:\jwlcool\TensorBoardX\PIDE')
    bg.reset()
    # for event in train_data.ordered_events:
    #     bg.add_event(event.user, event.item)

    model = DeepCoevolve(max_item_id,train_data,test_data,num_users=cmd_args.num_users,num_items=cmd_args.num_items,embed_size=cmd_args.embed_dim,
                         score_func=cmd_args.score_func, dt_type=cmd_args.dt_type, max_norm=cmd_args.max_norm,act=cmd_args.act).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate)

    cur_time.reset(0)
    for e in train_data.ordered_events:
        cur_time.update_event(e.user, e.item, e.t)    
    rc_dump = cur_time.dump()

    for epoch in range(cmd_args.num_epochs):
        if epoch==5:
            print("0")
        torch.cuda.empty_cache()
        cur_time.load_dump(*rc_dump)
        mar, HR_5, HR_10, HR_20, mae, rmse = model(kg,user_kg,relation_index,relation_dict,kg_user,train_data.ordered_events[-1].t,
                                                   test_data.ordered_events[:1000],  # 前1000个数据
                                                   phase='test')

        print('MAR:', mar, 'HR@5:', HR_5, 'HR@10:', HR_10, 'HR@20:', HR_20)
        sum_HR_5 += HR_5
        avg_HR_5 = sum_HR_5 / (epoch + 1)
        f = open(result, 'a')
        f.write(
            'MAR:  ' + str(mar) + ' ' + 'HR@5:  ' + str(HR_5) + ' ' + 'HR@10:  ' + str(HR_10) + ' ' + 'HR@20:  ' + str(
                HR_20))
        f.write('\n')
        f.close()
        writer.add_scalar('MAR', mar, global_step=epoch)
        pbar = tqdm(range(cmd_args.iters_per_val))

        Starttime = time.time()
        for it in pbar:
            cur_pos = np.random.randint(train_data.num_events - cmd_args.bptt)

            T_begin = 0
            if cur_pos:
                T_begin = train_data.ordered_events[cur_pos - 1].t
            
            event_mini_batch = train_data.ordered_events[cur_pos:cur_pos + cmd_args.bptt]
            
            optimizer.zero_grad()
            loss, mae, rmse = model(kg,user_kg,relation_index,relation_dict,kg_user,T_begin,event_mini_batch,phase='train')
            pbar.set_description('epoch: %.2f, loss: %.4f, mae: %.4f, rmse: %.4f' % (epoch + (it + 1) / len(pbar), loss.item(), mae, rmse))
            
            loss.backward()
            del loss,mae,rmse
            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)

            optimizer.step()
            model.normalize()
        Endtime = time.time()
        Calculate = Endtime - Starttime
        print('calculate:', Calculate)
        dur_dist.print_dist()







def read_item_index_to_entity_id_file(DATASET):
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1
    return  i

def convert_rating():
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
        # if user_index_old==274845:
        #     print('0')
        if DATASET=='Taobao' or DATASET=='Taobao_Big':
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
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
    user_list=list(set(user_list))
    item_list = list(set(item_list))
    user_list.sort()
    item_list.sort()
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
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
    cmd_args.num_users=user_cnt

    return user_index_old2new,item_index_old2new


def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0
    relation_dict=dict()
    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')
    for line in open('../data/' + DATASET + '/kg.txt', encoding='utf-8'):
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

    if cmd_args.kg_user and cmd_args.dataset=='movie-1m':
        user_file = '../data/' + DATASET + '/' + 'users.dat'
        for line in open(user_file, encoding='utf-8').readlines():
            array = line.strip().split('::')
            user = int(array[0])
            gender = array[1]
            age = int(array[2])
            occupation = int(array[3])
            user_id=user_index_old_new[user]
            if gender not in user_kg_gender2entity_id:
                # entity_id2index[gender] = entity_cnt
                user_kg_gender2entity_id[gender]=entity_cnt
                entity_cnt += 1
            gender_id = user_kg_gender2entity_id[gender]
            if age not in user_kg_age2entity_id:
                # entity_id2index[age] = entity_cnt
                user_kg_age2entity_id[age] = entity_cnt
                entity_cnt += 1
            age_id = user_kg_age2entity_id[age]
            if occupation not in user_kg_occupation2entity_id:
                # entity_id2index[occupation] = entity_cnt
                user_kg_occupation2entity_id[occupation] = entity_cnt
                entity_cnt += 1
            occupation_id = user_kg_occupation2entity_id[occupation]
            if user_id in user_kg: continue
            else:
                user_kg[user_id]=[gender_id,age_id,occupation_id]

    if cmd_args.kg_user and cmd_args.dataset == 'music':
        user_file = '../data/' + DATASET + '/' + 'user_friends.dat'
        for line in open(user_file, encoding='utf-8').readlines()[1:]:
            array = line.strip().split('\t')
            user = int(array[0])
            friend=int(array[1])
            if user not in user_index_old_new:continue
            if friend not in user_index_old_new: continue
            user_id=user_index_old_new[user]
            friend_id = user_index_old_new[friend]
            if user_id not in user_kg:
                user_kg[user_id]=[]
            user_kg[user_id].append(friend_id)
            user_kg[user_id]=list(set(user_kg[user_id]))
            relation_dict[(user_id, friend_id)] = -2
            relation_dict[(friend_id,user_id)] = -2



    if cmd_args.kg_user and cmd_args.dataset == 'Taobao':
        user_file = '../data/' + DATASET + '/' + 'small_user_profile_entity.csv'
        for line in open(user_file, encoding='utf-8').readlines()[:]:
            array = line.strip().split(',')
            user = int(array[0])
            if user not in user_index_old_new: continue
            user_id = user_index_old_new[user]
            # cms_segid= array[1]
            # cms_group_id= array[2]
            final_gender_code= array[3]
            age_level= array[4]
            pvalue_level= array[5]
            shopping_level= array[6]
            occupation= array[7]
            new_user_class_level= array[8]

            if final_gender_code!='':
                if final_gender_code not in user_kg_gender2entity_id:
                    entity_id2index[final_gender_code] = entity_cnt
                    user_kg_gender2entity_id[final_gender_code]=entity_cnt
                    entity_cnt += 1
                final_gender_code = entity_id2index[final_gender_code]

            if age_level!='':
                if age_level not in user_kg_age2entity_id:
                    entity_id2index[age_level] = entity_cnt
                    user_kg_age2entity_id[age_level]=entity_cnt
                    entity_cnt += 1
                age_level = entity_id2index[age_level]

            if pvalue_level!='':
                if pvalue_level not in user_kg_pvalue_level2entity_id:
                    entity_id2index[pvalue_level] = entity_cnt
                    user_kg_pvalue_level2entity_id[pvalue_level]=entity_cnt
                    entity_cnt += 1
                pvalue_level = entity_id2index[pvalue_level]

            if shopping_level!='':
                if shopping_level not in user_kg_shopping_level2entity_id:
                    entity_id2index[shopping_level] = entity_cnt
                    user_kg_shopping_level2entity_id[shopping_level]=entity_cnt
                    entity_cnt += 1
                shopping_level = entity_id2index[shopping_level]

            if occupation!='':
                if occupation not in user_kg_occupation2entity_id:
                    entity_id2index[occupation] = entity_cnt
                    user_kg_occupation2entity_id[occupation]=entity_cnt
                    entity_cnt += 1
                occupation = entity_id2index[occupation]

            if new_user_class_level!='':
                if new_user_class_level not in user_kg_user_class_level2entity_id:
                    entity_id2index[new_user_class_level] = entity_cnt
                    user_kg_user_class_level2entity_id[new_user_class_level]=entity_cnt
                    entity_cnt += 1
                new_user_class_level = entity_id2index[new_user_class_level]

            if user_id in user_kg:
                continue
            else:
                user_kg[user_id]=[final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level]

    # user_relation_dict = dict()
    if cmd_args.dataset=='Taobao' or cmd_args.dataset=='movie-1m':
        writer = open('../data/' + DATASET + '/kg_user_final.txt', 'w', encoding='utf-8')
        for line in open('../data/' + DATASET + '/kg_user.txt', encoding='utf-8'):
            array = line.strip().split('\t')
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]
            new_head = user_index_old_new[int(head_old)]
            new_tail = int(tail_old)


            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            writer.write('%d\t%d\t%d\n' % (new_head, relation, new_tail))

            relation_dict[(new_head, new_tail)] = relation

    writer.close()



    cmd_args.num_items = entity_cnt
    return relation_id2index,relation_dict,user_kg


def load_kg(args):
    kg=dict()
    kg_user=dict()

    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    if cmd_args.kg_user:
        if cmd_args.dataset=='Taobao' or cmd_args.dataset=='movie-1m':
            kg_user_file = '../data/' + args.dataset + '/kg_user_final'
            if os.path.exists(kg_user_file + '.npy'):
                kg_user_np = np.load(kg_user_file + '.npy')
            else:
                kg_user_np = np.loadtxt(kg_user_file + '.txt', dtype=np.int64)
                np.save(kg_user_file + '.npy', kg_user_np)
            n_entity = len(set(kg_user_np[:, 0]) | set(kg_user_np[:, 2]))
            n_relation = len(set(kg_user_np[:, 1]))
            kg_user = construct_kg(kg_user_np)


    return kg,kg_user,n_entity, n_relation

def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg










if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    # kg_test()
    entity_id2index = dict()
    relation_id2index = dict()
    relation_id2index['user.user'] = -2
    item_index_old2new=dict()
    max_item_id=read_item_index_to_entity_id_file(DATASET)
    user_index_old_new,item_index_old_new=convert_rating()
    relation_index,relation_dict,user_kg=convert_kg()

    kg,kg_user,n_entity, n_relation=load_kg(cmd_args)

    load_data()
    torch.backends.cudnn.benchmark = True
    # gc.enable()
    main_loop()
