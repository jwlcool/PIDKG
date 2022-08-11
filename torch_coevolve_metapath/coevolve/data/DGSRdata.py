dataset='music'

old_path='./'+dataset+'/all.txt'

new_path='./'+dataset+'/new.txt'


user_to_id=dict()
item_to_id=dict()
uid=0
iid=0
last=''
with open(new_path,'w') as f:
    f.write('user_id,item_id,time')
    f.write('\n')
    for line in open(old_path, encoding='utf-8').readlines()[:]:
        user_id = line.strip().split(' ')[0]
        item_id = line.strip().split(' ')[1]
        time = line.strip().split(' ')[2]

        if user_id not in user_to_id:
            user_to_id[user_id]=uid
            uid=uid+1
        user_id=user_to_id[user_id]
        if item_id not in item_to_id:
            item_to_id[item_id]=iid
            iid=iid+1
        item_id=item_to_id[item_id]


        sr=str(user_id)+','+str(item_id)+','+time
        if last==sr: continue
        last=sr
        f.write(sr)
        f.write('\n')
    f.close()




pass
