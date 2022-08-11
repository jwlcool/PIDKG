from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

import networkx as nx
from coevolve.common.cmd_args import cmd_args
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter


class Event(object):
    def __init__(self, user, item, t,rating, phase):
        self.user = user
        self.item = item
        self.t = t
        self.rating = rating
        self.phase = phase

        self.next_user_event = None
        self.prev_user_event = None
        self.prev_item_event = None
        self.global_idx = None


class Dataset(object):
    def __init__(self):
        self.user_event_lists = []
        self.item_event_lists = []
        self.ordered_events = []
        self.num_events = 0
        self.user_list = []
        self.item_list = []

    def load_events(self, ratio,filename, phase):
        self.user_event_lists = [[] for _ in range(cmd_args.num_users+1)]
        self.item_event_lists = [[] for _ in range(cmd_args.num_items+1)]

        count = -1
        for count, line in enumerate(open(filename, 'rU')):
            pass
        count += 1
        count=count*ratio
        with open(filename, 'r') as f:
            rows = f.readlines()
            for row in rows:
                user, item, t,rating = row.split()[:4]
                user = int(user)
                item = int(item)
                rating=float(rating)
                t = float(t) * cmd_args.time_scale
                cur_event = Event(user, item, t, rating,phase)
                self.ordered_events.append(cur_event)
                count=count-1
                if count<=-1:break

        # self.ordered_events = sorted(self.ordered_events, key=lambda x: x.t)
        for i in range(len(self.ordered_events)):
            cur_event = self.ordered_events[i]

            cur_event.global_idx = i
            user = cur_event.user
            item = cur_event.item
            self.user_list.append(user)
            self.item_list.append(item)

        self.user_list=list(set(self.user_list))
        self.item_list = list(set(self.item_list))
            # if user not in self.user_list:
            #     self.user_list.append(user)
            # if item not in self.item_list:
            #     self.item_list.append(item)

        for i in range(len(self.ordered_events)):
            cur_event = self.ordered_events[i]
            cur_event.global_idx = i
            user = cur_event.user
            item = cur_event.item
            if (user<cmd_args.num_users) and (item<cmd_args.num_items):

                if len(self.user_event_lists[user]):
                    cur_event.prev_user_event = self.user_event_lists[user][-1]
                if len(self.item_event_lists[item]):
                    cur_event.prev_item_event = self.item_event_lists[item][-1]
                if cur_event.prev_user_event is not None:
                    cur_event.prev_user_event.next_user_event = cur_event
                self.user_event_lists[user].append(cur_event)
                self.item_event_lists[item].append(cur_event)

        self.num_events = len(self.ordered_events)

    def clear(self):
        self.user_event_lists = []
        self.item_event_lists = []
        self.ordered_events = []

train_data = Dataset()
test_data = Dataset()



def create_kg(KG_entity,user_kg,kg,max_item_id,time_matrix,T_begain,user_event_list,item_event_list):

    for i in user_event_list:
        for j in i:
            ur='u'+str(j.user)
            im='i'+str(j.item)
            if j.t<=T_begain:
                kg.add_node(ur)
                kg.add_node(im)
                kg.add_edge(ur,im,time=j.t,rating=j.rating)
                if (j.user,j.item) not in time_matrix:
                    time_matrix[(j.user,j.item)] = j.t

                for k in KG_entity[j.item]:
                    e=k[0]
                    r=k[1]
                    entity='i'+str(e)
                    if cmd_args.kg_item and e>=max_item_id:
                        kg.add_node(entity)
                        kg.add_edge(im,entity,time=-1)

                if cmd_args.kg_user and cmd_args.dataset=='movie-1m':
                    if user_kg is not None:
                        for user_atr in user_kg[j.user]:
                            user_atr='i'+str(user_atr)
                            kg.add_node(user_atr)
                            kg.add_edge(ur, user_atr,time=-1)

                if cmd_args.kg_user and cmd_args.dataset=='Taobao':
                    if user_kg is not None:
                        for user_atr in user_kg[j.user]:
                            if user_atr!='':
                                user_atr='i'+str(user_atr)
                                kg.add_node(user_atr)
                                kg.add_edge(ur, user_atr,time=-1)

                if cmd_args.kg_user and cmd_args.dataset == 'music':
                    if user_kg is not None:
                        for user_friend in user_kg[j.user]:
                            user_atr='u'+str(user_friend)
                            kg.add_node(user_atr)
                            kg.add_edge(ur, user_atr,time=-1)
    return kg

def kg_add(kg,time_matrix,node1,node2,t):
    kg.add_node(node1)
    kg.add_node(node2)
    kg.add_edge(node1,node2,time=t)

    time_matrix[(int(node1[1:]),int(node2[1:]))]=t


def create_list(filename):
    user_list=[]
    item_list=[]
    with open(filename, 'r') as f:
        rows = f.readlines()
        for row in rows:
            user, item, t = row.split()[:3]
            user = int(user)
            item = int(item)
            if user not in user_list:
                user_list.append(user)
            if item not in item_list:
                item_list.append(item)
    return user_list,item_list

def merge_list(list1,list2):
    list=[]
    for i in list1 :
        if i not in list:
            list.append(i)
    for j in list2:
        if j not in list:
            list.append(j)
    list=sorted(list)
    return list

def get_neigbors(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output


def threshold(kg,time):
    for edge in kg.edges():
        remove=kg.get_edge_data(edge[0],edge[1])['time']
        if remove<time:
            kg.remove_edge(edge[0],edge[1])
    return kg

def get_edge_time(kg,h,t):
    time=kg.get_edge_data(h,t)['time']
    return time





def softmax(current_time,time_list):
    weight_agg={}
    weight = {}
    for source in time_list:
        weight[source]=[]
        exp_list = []
        sum_exp_x = 0
        for x in time_list[source]:
            x=x[0]
            exp_x = np.tanh(x/100)

            exp_list.append(exp_x)
            sum_exp_x = sum_exp_x+exp_x
        for id,x in enumerate(time_list[source]):
            if exp_list[id]/sum_exp_x<0.01:
                weight[source].append([x[1],0])
            else:
                weight[source].append([x[1],exp_list[id]/sum_exp_x])

    for i in weight:
        for j in weight[i]:
            if j[0] not in weight_agg:
                weight_agg[j[0]]={}
            if i not in weight_agg[j[0]]:
                weight_agg[j[0]][i]=j[1]
    return weight_agg

def softmax_time(current_time,time_list):
    weight_agg={}
    weight = {}
    for source in time_list:
        weight[source]=[]
        exp_list = []
        sum_exp_x = 0
        for x in time_list[source]:
            x=current_time-x[0]
            # x=-np.tanh(x/100000-10)+1
            exp_x = np.exp(-x/100000)
            if exp_x<=0.00000001:
                exp_x=0.00000001
            exp_list.append(exp_x)
            sum_exp_x = sum_exp_x+exp_x

        for id,x in enumerate(time_list[source]):
            weight[source].append([x[1],exp_list[id]/sum_exp_x])


    for i in weight:
        for j in weight[i]:
            if j[0] not in weight_agg:
                weight_agg[j[0]]={}
            if i not in weight_agg[j[0]]:
                weight_agg[j[0]][i]=j[1]
    return weight_agg

def weight_merge(time,path,ratio):
    weight=time
    weight_agg = {}
    for i in time:
        if i in path:
            for j in time[i]:
                time_weight=time[i][j]
                path_weight=path[i][j]
                weight[i][j]=time_weight*ratio+path_weight*(1-ratio)


    for i in weight:
        for j in weight[i]:
            if j not in weight_agg:
                weight_agg[j]={}
            if i not in weight_agg[j]:
                weight_agg[j][i]=weight[i][j]
    return weight_agg

def neighbor_ratio(p,ratio):
    weight=dict()
    weight_dict=dict()
    for i in p:
        if i not in weight:
            weight[i] = []
        keys=list(p[i].keys())
        values=list(p[i].values())
        for id,j in enumerate(keys):
            weight[i].append([values[id],keys[id]])
        weight[i].sort(reverse=True)
        neighbor_num = math.ceil(len(weight[i]) * (1 - ratio))
        weight[i] = weight[i][:neighbor_num]

    for i in weight:
        for j in weight[i]:
            if i not in weight_dict:
                weight_dict[i] = {}
            weight_dict[i][j[1]]=j[0]



    return weight_dict



def removeEmptyDict(data):
    data2 = {}
    for o in data:
        if not data[o] ==[]:
            data2[o] = data[o]
    return data2


def mine_paths_between_nodes(Graph,relation_index,relation_dict,user_relation_dict,user_node, location_node, maxLen,path_sample_size):
    lenth=0
    relation_count=0
    path_num=-1
    relation = -1
    path_relation_dict = dict()

    for path in nx.all_simple_paths(Graph, source=user_node, target=location_node, cutoff=maxLen):
        # if (int(user_node[1:]), int(location_node[1:])) not in path_relation_dict:
        #     path_relation_dict[(int(user_node[1:]), int(location_node[1:]))] = dict()
        path_num+=1

        if path_num not in path_relation_dict:
            path_relation_dict[path_num]=[]
        lenth = len(path)
        for id, node in enumerate(path):
            if id + 1 < lenth:
                if (int(node[1:]), int(path[id + 1][1:])) in relation_dict:

                    relation = relation_dict[(int(node[1:]), int(path[id + 1][1:]))]

                elif (int(node[1:]), int(path[id + 1][1:])) in user_relation_dict:

                    relation = relation_dict[(int(node[1:]), int(path[id + 1][1:]))]

                else:
                    # relation = 'user.item'
                    relation = -1

                path_relation_dict[path_num].extend([relation])
        if path_num>=path_sample_size-1:
            break

    relation_set=list()
    for j in range(path_num+1):
        relation_set.extend(path_relation_dict[j])
    relation_set=list(set(relation_set))

    similarity1_2 = 0
    similarity2_2 = 0
    similarity1_3 = 0
    similarity2_3 = 0
    similarity1_4 = 0
    similarity2_4 = 0
    count_2=0
    count_3=0
    count_4=0


    for i in relation_set:   #出现的每种关系
        for j in range(path_num+1): #每条路径上
            if len(path_relation_dict[j])+1==2:#路径长度为2的
                count_2+=1 #长度为2路径数量
                for n in path_relation_dict[j]:
                    if i==n:similarity1_2+=1

            if len(path_relation_dict[j])+1==3:
                count_3+=1
                for n in path_relation_dict[j]:
                    if i==n:similarity1_3+=1

            if len(path_relation_dict[j])+1==4:
                count_4+=1
                for n in path_relation_dict[j]:
                    if i==n:similarity1_4+=1

    similarity=similarity1_2/2+similarity1_3/3+similarity1_4/4

    count_2_sim1=0
    count_3_sim1=0
    count_3_sim2=0
    count_4_sim1=0
    count_4_sim2=0
    count_4_sim3=0
    sim_list2=list()
    sim_list3_1 = list()
    sim_list3_2 = list()
    sim_list4_1 = list()
    sim_list4_2 = list()
    sim_list4_3 = list()
    for j in range(path_num + 1):  # 每条路径上
        if len(path_relation_dict[j]) + 1 == 2:
            sim_list2.append(path_relation_dict[j][0])

        if len(path_relation_dict[j]) + 1 == 3:
            sim_list3_1.append(path_relation_dict[j][0])
            sim_list3_2.append(path_relation_dict[j][1])

        if len(path_relation_dict[j]) + 1 == 4:
            sim_list4_1.append(path_relation_dict[j][0])
            sim_list4_2.append(path_relation_dict[j][1])
            sim_list4_3.append(path_relation_dict[j][2])
    if len(sim_list2)!=0:
        count_2_sim1 = Counter(sim_list2).most_common(1)[0][1]

    if len(sim_list3_1) != 0:
        count_3_sim1 = Counter(sim_list3_1).most_common(1)[0][1]
    if len(sim_list3_2) != 0:
        count_3_sim2 = Counter(sim_list3_2).most_common(1)[0][1]

    count_3=(count_3_sim1+count_3_sim2)/2

    if len(sim_list4_1) != 0:
        count_4_sim1 = Counter(sim_list4_1).most_common(1)[0][1]
    if len(sim_list4_2) != 0:
        count_4_sim2 = Counter(sim_list4_2).most_common(1)[0][1]
    if len(sim_list4_3) != 0:
        count_4_sim3 = Counter(sim_list4_3).most_common(1)[0][1]
    count_4=(count_4_sim1+count_4_sim2+count_4_sim3)/3

    count=count_2_sim1+count_3+count_4

    if similarity==0 or count==0:
        value=0
    else:
        value=(path_num+1)*(path_num+1)/(1/similarity+1/count)
    return value




def dump_paths(Graph,relation_index, relation_dict,user_relation_dict, user_node,location_node, maxLen,path_sample_size):
    '''
    dump the postive or negative paths

    Inputs:
        @Graph: the well-built knowledge graph
        @rating_pair: positive_rating or negative_rating
        @maxLen: path length
        @sample_size: size of sampled paths between user-location nodes
    '''
    if Graph.has_node(user_node) and Graph.has_node(location_node):
        return mine_paths_between_nodes(Graph, relation_index, relation_dict,user_relation_dict,user_node, location_node, maxLen,path_sample_size)




def dump_path_list(Graph,relation_index, relation_dict, user_node,location_node, maxLen,path_sample_size):
    path_list=[]
    if Graph.has_node(user_node) and Graph.has_node(location_node):
        for path in nx.all_simple_paths(Graph, source=user_node, target=location_node, cutoff=maxLen):
            path_list.append(path)
            if len(path_list)>=path_sample_size:
                break
        return path_list


def path_value(path_list,relation_dict,user_relation_dict,user_node, location_node, maxLen):
    lenth=0
    relation_count=0
    path_num=-1
    relation = -1
    path_relation_dict = dict()
    for path in path_list:
        path_num += 1
        if path_num not in path_relation_dict:
            path_relation_dict[path_num] = []
        lenth = len(path)
        for id, node in enumerate(path):
            if id + 1 < lenth:
                if (int(node[1:]), int(path[id + 1][1:])) in relation_dict:

                    relation = relation_dict[(int(node[1:]), int(path[id + 1][1:]))]

                elif (int(node[1:]), int(path[id + 1][1:])) in user_relation_dict:

                    relation = relation_dict[(int(node[1:]), int(path[id + 1][1:]))]

                else:
                    # relation = 'user.item'
                    relation = -1

                path_relation_dict[path_num].extend([relation])

    relation_set=list()
    for j in range(path_num+1):
        relation_set.extend(path_relation_dict[j])
    relation_set=list(set(relation_set))

    similarity1_2 = 0
    similarity2_2 = 0
    similarity1_3 = 0
    similarity2_3 = 0
    similarity1_4 = 0
    similarity2_4 = 0
    count_2=0
    count_3=0
    count_4=0


    for i in relation_set:   #出现的每种关系
        for j in range(path_num+1): #每条路径上
            if len(path_relation_dict[j])+1==2:#路径长度为2的
                count_2+=1 #长度为2路径数量
                for n in path_relation_dict[j]:
                    if i==n:similarity1_2+=1

            if len(path_relation_dict[j])+1==3:
                count_3+=1
                for n in path_relation_dict[j]:
                    if i==n:similarity1_3+=1

            if len(path_relation_dict[j])+1==4:
                count_4+=1
                for n in path_relation_dict[j]:
                    if i==n:similarity1_4+=1

    similarity=similarity1_2/2+similarity1_3/3+similarity1_4/4

    count_2_sim1=0
    count_3_sim1=0
    count_3_sim2=0
    count_4_sim1=0
    count_4_sim2=0
    count_4_sim3=0
    sim_list2=list()
    sim_list3_1 = list()
    sim_list3_2 = list()
    sim_list4_1 = list()
    sim_list4_2 = list()
    sim_list4_3 = list()
    for j in range(path_num + 1):  # 每条路径上
        if len(path_relation_dict[j]) + 1 == 2:
            sim_list2.append(path_relation_dict[j][0])

        if len(path_relation_dict[j]) + 1 == 3:
            sim_list3_1.append(path_relation_dict[j][0])
            sim_list3_2.append(path_relation_dict[j][1])

        if len(path_relation_dict[j]) + 1 == 4:
            sim_list4_1.append(path_relation_dict[j][0])
            sim_list4_2.append(path_relation_dict[j][1])
            sim_list4_3.append(path_relation_dict[j][2])
    if len(sim_list2)!=0:
        count_2_sim1 = Counter(sim_list2).most_common(1)[0][1]

    if len(sim_list3_1) != 0:
        count_3_sim1 = Counter(sim_list3_1).most_common(1)[0][1]
    if len(sim_list3_2) != 0:
        count_3_sim2 = Counter(sim_list3_2).most_common(1)[0][1]

    count_3=(count_3_sim1+count_3_sim2)/2

    if len(sim_list4_1) != 0:
        count_4_sim1 = Counter(sim_list4_1).most_common(1)[0][1]
    if len(sim_list4_2) != 0:
        count_4_sim2 = Counter(sim_list4_2).most_common(1)[0][1]
    if len(sim_list4_3) != 0:
        count_4_sim3 = Counter(sim_list4_3).most_common(1)[0][1]
    count_4=(count_4_sim1+count_4_sim2+count_4_sim3)/3

    count=count_2_sim1+count_3+count_4



    if similarity==0 or count==0:
        value=0
    else:
        value=(path_num+1)/(1/similarity+1/count)
    return value



