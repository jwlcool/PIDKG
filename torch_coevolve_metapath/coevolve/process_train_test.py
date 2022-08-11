small_user='./data/Taobao/small_user_list.csv'
small_item='./data/Taobao/small_item_list.csv'
small_userprofile='./data/Taobao/small_user_profile.csv'
small_ratingpath='./data/Taobao/small_raw_sample.csv'
small_ratingpath_order='./data/Taobao/small_raw_sample_order.csv'
small_itemfeature='./data/Taobao/small_ad_feature.csv'

small_ratingpath_click='./data/Taobao/small_raw_sample_click.csv'
entity_rating='./data/Taobao/rating_entity.csv'
train_path='./data/Taobao/train.txt'
test_path='./data/Taobao/test.txt'
meta_path='./data/Taobao/meta.txt'
final_rating='./data/Taobao/rating.csv'
train_ratio=0.7
rating_len=0

for line in open(entity_rating, encoding='utf-8').readlines()[1:]:
    rating_len+=1

print("Rating len:")
print(rating_len)
user_list=[]
item_list=[]
train_count=0
train_writer = open(train_path, 'w', encoding='utf-8')
test_writer = open(test_path, 'w', encoding='utf-8')
for line in open(entity_rating, encoding='utf-8').readlines()[1:]:
    train_count += 1
    user_id = line.strip().split('\t')[0]
    time_stamp=line.strip().split('\t')[3]
    item_id = line.strip().split('\t')[1]
    user_list.append(int(line.strip().split('\t')[0]))
    item_list.append(int(line.strip().split('\t')[1]))
    s=user_id+" "+item_id+" "+time_stamp+" "+"1 0 0"
    if train_count <= rating_len * train_ratio:
        train_writer.write(s)
        train_writer.write('\n')
    if train_count > rating_len * train_ratio:
        test_writer.write(s)
        test_writer.write('\n')
user_list=list(set(user_list))
item_list=list(set(item_list))
print("user num:")
print(len(user_list))
print("item num:")
print(len(item_list))
meta_writer= open(meta_path, 'w', encoding='utf-8')
s=str(len(user_list)+1)+" "+str(len(item_list)+1)+" "+"1"
meta_writer.write(s)





