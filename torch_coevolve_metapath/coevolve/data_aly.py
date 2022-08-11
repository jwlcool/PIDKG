small_ratingpath_click='./data/Taobao/small_raw_sample_click.csv'
small_ratingpath_order='./data/Taobao/small_raw_sample_order.csv'

user_rating_list=[]
user_history_click=dict()
user_history_order=dict()
item_history=dict()
for line in open(small_ratingpath_click, encoding='utf-8').readlines()[1:]:
    user_id = int(line.strip().split(',')[0])
    time_stamp=int(line.strip().split(',')[1])
    item_id = int(line.strip().split(',')[2])
    user_rating_list.append(user_id)
    if user_id not in user_history_click:
        user_history_click[user_id] = 0
    if user_id in user_history_click:
        user_history_click[user_id]+=1

for line in open(small_ratingpath_order, encoding='utf-8').readlines()[1:]:
    user_id = int(line.strip().split(',')[0])
    time_stamp=int(line.strip().split(',')[1])
    item_id = int(line.strip().split(',')[2])
    user_rating_list.append(user_id)
    if user_id not in user_history_order:
        user_history_order[user_id] = 0
    if user_id in user_history_order:
        user_history_order[user_id]+=1

print('0')