import numpy as np
testfile=open('logistic_sub.csv')
testfile.readline()
testdata=[]
id=[]
ii=[]
for line in testfile:
    list=line.rstrip().split(',')
    ii.append(int(list[0]))
    id.append(float(list[1]))

testfile=open('nb_predict.csv')
testfile.readline()
testdata=[]
id2=[]
for line in testfile:
    list=line.rstrip().split(',')
    id2.append(float(list[1]))

id=np.array(id)
id2=np.array(id2)
print(id)
print(id2)
id=0.5*id
id2=0.5*id2
print(id)
print(id2)
id3=id+id2

i=0
submitfile=open('submission2.csv','w')
submitfile.write('Id,Action\n')
while i<len(id):
    submitfile.write(str(ii[i])+","+str(id3[i])+"\n")
    i=i+1
