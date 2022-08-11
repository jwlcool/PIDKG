ratingpath='./data/Taobao/raw_sample/raw_sample.csv'
itemfeature='./data/Taobao/ad_feature/ad_feature.csv'
userfeature='./data/Taobao/user_profile/user_profile.csv'

small_user='./data/Taobao/small_user_list.csv'
small_item='./data/Taobao/small_item_list.csv'
item_2entity='./data/Taobao/item_index2entity_id.txt'
small_userprofile='./data/Taobao/small_user_profile.csv'
small_ratingpath='./data/Taobao/small_raw_sample.csv'
small_ratingpath_order='./data/Taobao/small_raw_sample_order.csv'
small_itemfeature='./data/Taobao/small_ad_feature.csv'

small_ratingpath_click='./data/Taobao/small_raw_sample_click.csv'


user_list=[]
small_user_list=[]
item_list=[]
small_item_list=[]
user_history=dict()
item_history=dict()
user_dict=dict()
item_dict=dict()
user_num=2000

##提取部分用户列表

for line in open(userfeature, encoding='utf-8').readlines()[1:]:
    user_id = line.strip().split(',')[0]
    user_list.append(int(user_id))

for line in open(itemfeature, encoding='utf-8').readlines()[1:]:
    item_id = line.strip().split(',')[0]
    item_list.append(int(item_id))

user_list=list(set(user_list))
item_list=list(set(item_list))
user_list.sort()
item_list.sort()

for id,value in enumerate(user_list):
    user_history[value]=0


user_rating_list=[]
for line in open(ratingpath, encoding='utf-8').readlines()[1:]:
    user_id = int(line.strip().split(',')[0])
    time_stamp=int(line.strip().split(',')[1])
    item_id = int(line.strip().split(',')[2])
    user_rating_list.append(user_id)

    if user_id not in user_history:
        user_history[user_id]=0
    if user_id in user_history:
        user_history[user_id]+=1



user_rating_list=list(set(user_rating_list))
user_rating_list.sort()


k=sorted(user_history.items(), key = lambda x:(x[1]),reverse=True)
small_user_dict=k[1000:1000+user_num]
small_user_list=[]
writer = open(small_user, 'w', encoding='utf-8')
for key in small_user_dict:
    small_user_list.append(key[0])
    writer.write(str(key[0]))
    writer.write('\n')
# k=user_rating_list
# small_user_dict=k[:user_num]
# small_user_list=[]
# writer = open(small_user, 'w', encoding='utf-8')
# for key in small_user_dict:
#     small_user_list.append(key)
#     writer.write(str(key))
#     writer.write('\n')



#####根据用户列表提取项目列表，生成较小的交互文件
writer = open(small_ratingpath, 'w', encoding='utf-8')
small_item_list=[]
lengh=0
for line in open(ratingpath, encoding='utf-8').readlines()[1:]:
    user_id = int(line.strip().split(',')[0])
    time_stamp=int(line.strip().split(',')[1])
    item_id = int(line.strip().split(',')[2])
    if user_id in small_user_list:
        small_item_list.append(item_id)
        writer.write(line)
        lengh+=1
##############交互文件排序
path=open(small_ratingpath)
new_path=small_ratingpath_order
result=[]
iter_f=iter(path)
for line in iter_f:
    if line[0]=='u':continue
    result.append(line)
path.close()
result.sort(key=lambda x:float(x.split(',')[1]),reverse=False)
f=open(new_path,'w')
f.write('\n')
f.writelines(result)
f.close()


##########筛选出点击过的交互
item_list=[]
user_list=[]
click_writer=open(small_ratingpath_click, 'w', encoding='utf-8')
for line in open(small_ratingpath_order, encoding='utf-8').readlines()[1:]:
    user_id=int(line.strip().split(',')[0])
    item_id = int(line.strip().split(',')[2])
    click=int(line.strip().split(',')[5])
    if click==1:
        click_writer.write(line)
        item_list.append(item_id)
        user_list.append(user_id)

item_list=list(set(item_list))
item_list.sort()
user_list=list(set(user_list))
user_list.sort()
print("Clicked item:")
print(len(item_list))
click_writer.close()

with open(small_item,'w') as f:
    for i in item_list:
        f.write(str(i))
        f.write('\n')
f.close()

with open(small_user,'w') as f:
    for i in user_list:
        f.write(str(i))
        f.write('\n')
f.close()

writer = open(small_userprofile, 'w', encoding='utf-8')
for line in open(userfeature, encoding='utf-8').readlines()[1:]:
    user_id = int(line.strip().split(',')[0])
    if user_id in user_list:
        writer.write(line)
writer.close()

writer = open(small_itemfeature, 'w', encoding='utf-8')
for line in open(itemfeature, encoding='utf-8').readlines()[1:]:
    item_id = int(line.strip().split(',')[0])
    if item_id in item_list:
        writer.write(line)
writer.close()






######生成 item_2entity 文件
# with open(item_2entity,'w') as f:
#     for id,value in enumerate(small_item_list):
#         s=str(value)+"\t"+str(id)
#         f.write(s)
#         f.write('\n')
# print("Rating len: ")
# print(len)
# writer.close()














