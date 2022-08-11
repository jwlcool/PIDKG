ratingpath='./data/Taobao_Big/raw_sample/raw_sample.csv'
itemfeature='./data/Taobao_Big/ad_feature/ad_feature.csv'
userfeature='./data/Taobao_Big/user_profile/user_profile.csv'
small_user='./data/Taobao_Big/small_user_list.csv'
small_item='./data/Taobao_Big/small_item_list.csv'
small_userprofile='./data/Taobao_Big/small_user_profile.csv'
small_userprofile_entity='./data/Taobao_Big/small_user_profile_entity.csv'
small_ratingpath='./data/Taobao_Big/small_raw_sample.csv'
small_ratingpath_order='./data/Taobao_Big/small_raw_sample_order.csv'
small_itemfeature='./data/Taobao_Big/small_ad_feature.csv'

item_2entity='./data/Taobao_Big/item_index2entity_id.txt'
cate_2entity='./data/Taobao_Big/cate_index2entity_id.txt'
campaign_2entity='./data/Taobao_Big/campaign_index2entity_id.txt'
brand_2entity='./data/Taobao_Big/brand_index2entity_id.txt'
price_2entity='./data/Taobao_Big/price_index2entity_id.txt'

small_ratingpath_click='./data/Taobao_Big/small_raw_sample_click.csv'
final_rating='./data/Taobao_Big/rating.csv'
entity_rating='./data/Taobao_Big/rating_entity.csv'
item_kg='./data/Taobao_Big/kg.txt'

user_kg='./data/Taobao_Big/kg_user.txt'

cate_node=[]
p='0'


item_list=[]
cate_list=[]
campaign_list=[]
customer_list=[]
brand_list=[]
price_list=[]
#
# user_list=[]
# items_list=[]
#
# for line in open(small_ratingpath_click, encoding='utf-8').readlines()[1:]:
#     user_id = int(line.strip().split(',')[0])
#     time_stamp=int(line.strip().split(',')[1])
#     item_id = int(line.strip().split(',')[2])
#
#     user_list.append(user_id)
#     items_list.append(item_id)
#
# user_list=list(set(user_list))
# items_list=list(set(items_list))
# user_list.sort()
# items_list.sort()

rating_writer=open(final_rating, 'w', encoding='utf-8')
rating_writer.write('u,i,rating,time')
rating_writer.write('\n')
for line in open(small_ratingpath_click, encoding='utf-8').readlines()[1:]:
    user_id = line.strip().split(',')[0]
    time_stamp=line.strip().split(',')[1]
    item_id = line.strip().split(',')[2]
    s=user_id+'\t'+item_id+'\t'+'NULL'+'\t'+time_stamp
    rating_writer.write(s)
    rating_writer.write('\n')
rating_writer.close()


for line in open(small_itemfeature, encoding='utf-8').readlines()[:]:
    item_id = int(line.strip().split(',')[0])
    cate_id= int(line.strip().split(',')[1])
    campaign_id= int(line.strip().split(',')[2])
    customer= int(line.strip().split(',')[3])
    if line.strip().split(',')[4]!='NULL':
        brand= int(line.strip().split(',')[4])
        brand_list.append(brand)

    item_list.append(item_id)
    cate_list.append(cate_id)
    campaign_list.append(campaign_id)
    customer_list.append(customer)
    price_list=[1,2,3,4,5]

    item_list=list(set(item_list))
    cate_list = list(set(cate_list))
    campaign_list=list(set(campaign_list))
    customer_list = list(set(customer_list))
    brand_list = list(set(brand_list))
    item_list.sort()
    cate_list.sort()
    campaign_list.sort()
    customer_list.sort()
    brand_list.sort()
# #
#
item_dict=dict()
cate_dict=dict()
campaign_dict=dict()
brand_dict=dict()
price_dict=dict()
#
entity_count=0
entity_dict=dict()
with open(item_2entity,'w') as f:
    for id,value in enumerate(item_list):

        item_dict[value]=entity_count
        s=str(value)+"\t"+str(entity_count)
        entity_count += 1
        f.write(s)
        f.write('\n')

with open(cate_2entity,'w') as f:
    for id,value in enumerate(cate_list):

        cate_dict[value]=entity_count
        s=str(value)+"\t"+str(entity_count)
        entity_count += 1
        f.write(s)
        f.write('\n')

with open(campaign_2entity,'w') as f:
    for id,value in enumerate(campaign_list):

        campaign_dict[value] = entity_count
        s=str(value)+"\t"+str(entity_count)
        entity_count += 1
        f.write(s)
        f.write('\n')

with open(brand_2entity,'w') as f:
    for id,value in enumerate(brand_list):

        brand_dict[value] = entity_count
        s=str(value)+"\t"+str(entity_count)
        entity_count += 1
        f.write(s)
        f.write('\n')

with open(price_2entity,'w') as f:
    for id,value in enumerate(price_list):

        price_dict[value] = entity_count
        s=str(value)+"\t"+str(entity_count)
        entity_count += 1
        f.write(s)
        f.write('\n')



entityid_writer=open(entity_rating, 'w', encoding='utf-8')
entityid_writer.write('u,i,rating,time')
entityid_writer.write('\n')
user_index_old2new = dict()
user_pos_ratings = dict()
user_neg_ratings = dict()
file=final_rating
user_cnt = 0
for line in open(file, encoding='utf-8').readlines()[1:]:
    array = line.strip().split('\t')
    user_index_old = int(array[0])
    if user_index_old not in user_index_old2new:
        user_index_old2new[user_index_old] = user_cnt
        user_cnt += 1
    user_index = user_index_old2new[user_index_old]

for line in open(small_ratingpath_click, encoding='utf-8').readlines()[1:]:
    user_id = int(line.strip().split(',')[0])
    time_stamp=line.strip().split(',')[1]
    item_id = int(line.strip().split(',')[2])

    new_user_id=user_index_old2new[user_id]
    new_item_id=item_dict[item_id]


    s=str(new_user_id)+'\t'+str(new_item_id)+'\t'+'NULL'+'\t'+time_stamp

    entityid_writer.write(s)
    entityid_writer.write('\n')
entityid_writer.close()


gender_dict=dict()
age_dict=dict()
pvalue_level_dict=dict()
shopping_level_dict=dict()
occupation_dict=dict()
new_user_class_level_dict=dict()
userkg_writer = open(user_kg, 'w', encoding='utf-8')
user_profile_entiy_writer=open(small_userprofile_entity, 'w', encoding='utf-8')
for line in open(small_userprofile, encoding='utf-8').readlines()[:]:
    user_id = int(line.strip().split(',')[0])

    cms_segid=line.strip().split(',')[1]
    cms_group_id=line.strip().split(',')[2]

    gender_id=line.strip().split(',')[3]
    if gender_id not in gender_dict:
        gender_dict[gender_id]=entity_count
        entity_count+=1
    new_gender_id=gender_dict[gender_id]
    gender_id_relation = str(user_id) + "\t" + "user.gender" + "\t" + str(new_gender_id)
    userkg_writer.write(gender_id_relation)
    userkg_writer.write('\n')

    age_id=line.strip().split(',')[4]
    if age_id not in age_dict:
        age_dict[age_id]=entity_count
        entity_count+=1
    new_age_id=age_dict[age_id]
    age_id_relation = str(user_id) + "\t" + "user.age" + "\t" + str(new_age_id)
    userkg_writer.write(age_id_relation)
    userkg_writer.write('\n')

    pvalue_level=line.strip().split(',')[5]
    if pvalue_level!='':
        if pvalue_level not in pvalue_level_dict:
            pvalue_level_dict[pvalue_level]=entity_count
            entity_count+=1
        new_pvalue_level=pvalue_level_dict[pvalue_level]
        pvalue_level_relation = str(user_id) + "\t" + "user.pvalue_level" + "\t" + str(new_pvalue_level)
        userkg_writer.write(pvalue_level_relation)
        userkg_writer.write('\n')
    else:
        new_pvalue_level=pvalue_level

    shopping_level=line.strip().split(',')[6]
    if shopping_level!='':
        if shopping_level not in shopping_level_dict:
            shopping_level_dict[shopping_level]=entity_count
            entity_count+=1
        new_shopping_level=shopping_level_dict[shopping_level]
        shopping_level_relation = str(user_id) + "\t" + "user.shopping_level" + "\t" + str(new_shopping_level)
        userkg_writer.write(shopping_level_relation)
        userkg_writer.write('\n')
    else:
        new_shopping_level =shopping_level

    occupation=line.strip().split(',')[7]
    if occupation!='':
        if occupation not in occupation_dict:
            occupation_dict[occupation]=entity_count
            entity_count+=1
        new_occupation_id=occupation_dict[occupation]
        occupation_id_relation = str(user_id) + "\t" + "user.occupation" + "\t" + str(new_occupation_id)
        userkg_writer.write(occupation_id_relation)
        userkg_writer.write('\n')
    else:
        new_occupation_id =occupation

    new_user_class_level =line.strip().split(',')[8]
    if new_user_class_level != '':
        if new_user_class_level not in new_user_class_level_dict:
            new_user_class_level_dict[new_user_class_level]=entity_count
            entity_count+=1
        new_new_user_class_level=new_user_class_level_dict[new_user_class_level]
        user_class_level_relation = str(user_id) + "\t" + "user.user_class_level" + "\t" + str(new_new_user_class_level)
        userkg_writer.write(user_class_level_relation)
        userkg_writer.write('\n')
    else:
        new_new_user_class_level =new_user_class_level

    new_s=str(user_id)+','+cms_segid+','+cms_group_id+','+str(new_gender_id)+','+str(new_age_id)+','+\
          str(new_pvalue_level) +','+str(new_shopping_level)+','+str(new_occupation_id)+','+str(new_new_user_class_level)

    user_profile_entiy_writer.write(new_s)
    user_profile_entiy_writer.write('\n')

    # cms_segid_relation=str(user_id)+"\t"+"user.cms_segid"+"\t"+cms_segid
    # cms_group_id_relation = str(user_id) + "\t" + "user.cms_segid" + "\t" + cms_group_id





item_writer = open(item_kg, 'w', encoding='utf-8')
for line in open(small_itemfeature, encoding='utf-8').readlines()[:]:
    item_id = int(line.strip().split(',')[0])
    item_id=item_dict[item_id]
    cate_id = int(line.strip().split(',')[1])
    cate_id = cate_dict[cate_id]
    campaign_id= int(line.strip().split(',')[2])
    campaign_id = campaign_dict[campaign_id]
    # customer_id= int(line.strip().split(',')[3])
    if line.strip().split(',')[4]!="NULL":
        brand_id= int(line.strip().split(',')[4])
        brand_id = brand_dict[brand_id]
    price_id= float(line.strip().split(',')[5])

    if price_id>0 and price_id<=100:
        p=1
    if price_id>100 and price_id<=300:
        p=2
    if price_id>300 and price_id<=500:
        p=3
    if price_id>500 and price_id<=1000:
        p=4
    if price_id>1000:
        p=5
    price_id = price_dict[p]

    cate=str(item_id)+"\t"+"item.category"+"\t"+str(cate_id)
    campaign=str(item_id)+"\t"+"item.campaign"+"\t"+str(campaign_id)
    if line.strip().split(',')[4] != "NULL":
        brand=str(item_id)+"\t"+"item.brand"+"\t"+str(brand_id)
        item_writer.write(brand)
        item_writer.write('\n')
    price=str(item_id)+"\t"+"item.price"+"\t"+str(price_id)

    item_writer.write(cate)
    item_writer.write('\n')
    item_writer.write(campaign)
    item_writer.write('\n')
    item_writer.write(price)
    item_writer.write('\n')





