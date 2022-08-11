dataset='Taobao'
ratio=0.7
test_ratio=0.3
dataset_all=dataset+'/all.txt'

train_file=dataset+'/train'+str(ratio)+'.txt'
test_file=dataset+'/test'+str(ratio)+'.txt'
writer = open(train_file, 'w', encoding='utf-8')
test_writer=open(test_file, 'w', encoding='utf-8')

count=0
for line in open(dataset_all, encoding='utf-8').readlines()[:]:
    count+=1


train_size=ratio*count
test_size=test_ratio*count
print(train_size)
print(test_size)

flag=0
for line in open(dataset_all, encoding='utf-8').readlines()[:]:
    if flag<train_size:
        writer.write(line)
        flag+=1
    else:
        if flag < train_size+test_size:
            test_writer.write(line)
            flag += 1

