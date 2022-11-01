import statistics
import numpy as np
import pandas as pd
from random import random
import pickle
featureClass=pd.read_csv('featureclass_3.csv')
featureClass=featureClass.to_numpy()
featureClass=np.delete(featureClass,0,1)
#fc=[]
#for x in range(0,len(featureClass)):
    #fc.append(featureClass[x][0])
#featureClass=fc.copy()
feats3=pd.read_csv('feats_3.csv')
target=feats3["64"]
feats2=feats3.drop(["64"],axis=1)
feats2=feats2.to_numpy()
feats2=np.delete(feats2,0,1)
#feats2=feats2[len(feats2)-5000:len(feats2),:]
#target=target[len(feats2)-5000:len(feats2)]
target=target.to_numpy()
feats3=feats3.to_numpy()
feats3=np.delete(feats3,0,1)
nl=60
weights1=np.zeros((nl,feats2.shape[1]))
biases1=np.zeros(nl)
layer2s=[]
outs=[]
print(random())
for x in range(0,len(weights1[0])):
    for y in range(0,len(weights1)):
        r=random()
        if r<0.34:
            weights1[y][x]=-0.1
        elif r<0.67:
            weights1[y][x]=0
        else:
            weights1[y][x]=0.1
for n in range(0,len(biases1)):
    r=random()
    if r<0.34:
        biases1[n]=-0.1
    elif r<0.67:
        biases1[n]=0
    else:
        biases1[n]=0.1
weights2=np.zeros((featureClass.shape[0],nl))
biases2=np.zeros(featureClass.shape[0])
for x in range(0,len(weights2[0])):
    for y in range(0,len(weights2)):
        r=random()
        if r<0.34:
            weights2[y][x]=-0.1
        elif r<0.67:
            weights2[y][x]=0
        else:
            weights2[y][x]=0.1
for n in range(0,len(biases2)):
    r=random()
    if r<0.34:
        biases2[n]=-0.1
    elif r<0.67:
        biases2[n]=0
    else:
        biases2[n]=0.1
testers1=[weights1.copy(),weights1.copy(),weights1.copy(),weights1.copy(),weights1.copy(),weights1.copy(),weights1.copy(),weights1.copy()]
testers2=[weights2.copy(),weights2.copy(),weights2.copy(),weights2.copy(),weights2.copy(),weights2.copy(),weights2.copy(),weights2.copy()]
btesters1=[biases1.copy(),biases1.copy(),biases1.copy(),biases1.copy(),biases1.copy(),biases1.copy(),biases1.copy(),biases1.copy()]
btesters2=[biases2.copy(),biases2.copy(),biases2.copy(),biases2.copy(),biases2.copy(),biases2.copy(),biases2.copy(),biases2.copy()]
for x in range(0,len(testers1[4][0])):
    for y in range(0,len(testers1[4])):
        if testers1[0][y][x]==0.1:
            testers1[4][y][x]=-0.1
        elif testers1[0][y][x]==-0.1:
            testers1[4][y][x]=0.1
for x in range(0,len(testers2[4][0])):
    for y in range(0,len(testers2[4])):
        if testers2[0][y][x]==0.1:
            testers2[4][y][x]=-0.1
        elif testers2[0][y][x]==-0.1:
            testers2[4][y][x]=0.1
for n in range(0,len(btesters1[4])):
    if btesters1[4][n]==0.1:
        btesters1[4][n]=-0.1
    elif btesters1[4][n]==-0.1:
        btesters1[4][n]=0.1
for n in range(0,len(btesters2[4])):
    if btesters2[4][n]==0.1:
        btesters2[4][n]=-0.1
    elif btesters2[4][n]==-0.1:
        btesters2[4][n]=0.1
scores=[]
for t in range(0,8):
    if t>3:
        tt=t-4
    else:
        tt=t
    for x in range(0,len(testers1[t][0])):
        for y in range(0,len(testers1[t])):
            if random()<tt*0.25:
                if testers1[t][y][x]==0.1:
                    if random()<0.5:
                        testers1[t][y][x]=-0.1
                    else:
                        testers1[t][y][x]=0
                elif testers1[t][y][x]==-0.1:
                    if random()<0.5:
                        testers1[t][y][x]=0.1
                    else:
                        testers1[t][y][x]=0
                else:
                    if random()<0.5:
                        testers1[t][y][x]=-0.1
                    else:
                        testers1[t][y][x]=0.1
    for x in range(0,len(testers2[t][0])):
        for y in range(0,len(testers2[t])):
            if random()<tt*0.25:
                if testers2[t][y][x]==0.1:
                    if random()<0.5:
                        testers2[t][y][x]=-0.1
                    else:
                        testers2[t][y][x]=0
                elif testers2[t][y][x]==-0.1:
                    if random()<0.5:
                        testers2[t][y][x]=0.1
                    else:
                        testers2[t][y][x]=0
                else:
                    if random()<0.5:
                        testers2[t][y][x]=-0.1
                    else:
                        testers2[t][y][x]=0.1
    for x in range(0,len(btesters1[t])):
        if random()<tt*0.25:
            if btesters1[t][x]==0.1:
                if random()<0.5:
                    btesters1[t][x]=-0.1
                else:
                    btesters1[t][x]=0
            elif btesters1[t][x]==-0.1:
                if random()<0.5:
                    btesters1[t][x]=0.1
                else:
                    btesters1[t][x]=0
            else:
                if random()<0.5:
                    btesters1[t][x]=-0.1
                else:
                    btesters1[t][x]=0.1
    for x in range(0,len(btesters2[t])):
        if random()<tt*0.25:
            if btesters2[t][x]==0.1:
                if random()<0.5:
                    btesters2[t][x]=-0.1
                else:
                    btesters2[t][x]=0
            elif btesters2[t][x]==-0.1:
                if random()<0.5:
                    btesters2[t][x]=0.1
                else:
                    btesters2[t][x]=0
            else:
                if random()<0.5:
                    btesters2[t][x]=-0.1
                else:
                    btesters2[t][x]=0.1
    score=0
    for n in range(0,len(feats2)):
        layer2=np.dot(testers1[t],feats2[n,:])+btesters1[t]
        for x in range(0,len(layer2)):
            #layer2[x]=1/(1+np.exp(-layer2[x]))
            if layer2[x]<0:
                layer2[x]=0
        out=np.dot(testers2[t],layer2)+btesters2[t]
        if np.argmax(out)==target[n]:
            score+=1
    print(score)
    scores.append(score)
amax=scores.index(max(scores))
themax=scores[amax]
print(themax)
#themax2=nl0000000
weights1=testers1[amax]
weights2=testers2[amax]
biases1=btesters1[amax]
biases2=btesters2[amax]
'''
weights1=pd.read_csv('weights_1.csv')
weights1=weights1.to_numpy()
weights1=np.delete(weights1,0,1)
weights2=pd.read_csv('weights_2.csv')
weights2=weights2.to_numpy()
weights2=np.delete(weights2,0,1)
biases1=pd.read_csv('biases_1.csv')
biases1=biases1.to_numpy()
biases1=np.delete(biases1,0,1)
biases2=pd.read_csv('biases_2.csv')
biases2=biases2.to_numpy()
biases2=np.delete(biases2,0,1)
biases1=biases1[:,0]
biases2=biases2[:,0]
with open('thescore.txt', "rb") as fp:
    thescore=pickle.load(fp)
print(thescore)
'''
y=0
yy=0
st=1
ts=1
stopper=30
thescaler=1
nf=len(feats2)
#with open("thescaler.txt", "wb") as fp:   #Pickling
    #pickle.dump(thescaler, fp)
scores=[]
while True:
    grad1=np.zeros((nl,feats2.shape[1]))
    grad2=np.zeros((featureClass.shape[0],nl))
    gb2=np.zeros(featureClass.shape[0])
    gb1=np.zeros(nl)
    score=0
    for n in range(0,len(feats2)):
        layer2=np.dot(weights1,feats2[n,:])+biases1
        for x in range(0,len(layer2)):
            '''
            ee=thescaler*layer2[x]
            if ee>100:
                ee=100
            elif ee<-100:
                ee=-100
            #layer2[x]=1/(1+np.exp(-ee))
            '''
            if layer2[x]<0:
                layer2[x]=0
            #'''
        out=np.dot(weights2,layer2)+biases2
        for x in range(0,len(out)):
            #out[x]=1/(1+np.exp(-out[x]))
            if out[x]<0:
                out[x]=0
            #print(out[x])
        m=np.argmax(out)
        if m!=target[n]:
            #while m!=target[n]:
            #print('got'+featureClass[m][0])
            #print('should be'+featureClass[int(target[n])][0])
            dif=out[m]-out[int(target[n])]
            if dif<0.01:
                dif=0.01
            if stopper==30:
                gb2[int(target[n])]+=2*(1/nf)*(dif)#*(out[int(target[n])]*(1-out[int(target[n])]))
                grad2[int(target[n])]+=2*(1/nf)*layer2*(dif)#*(out[int(target[n])]*(1-out[int(target[n])]))
                gb2[m]+=2*(1/nf)*(-dif)#*(out[m]*(1-out[m]))
                grad2[m]+=2*(1/nf)*layer2*(-dif)#*(out[m]*(1-out[m]))
            else:
                for h in range(0,nl):
                    grad1[h]+=feats2[n,:]*2*(1/nf)*weights2[int(target[n])][h]*(dif)#*(layer2[h]*(1-layer2[h]))
                    gb1[h]+=2*(1/nf)*weights2[int(target[n])][h]*(dif)#*(layer2[h]*(1-layer2[h]))
                    grad1[h]+=feats2[n,:]*2*(1/nf)*weights2[m][h]*(-dif)#*(layer2[h]*(1-layer2[h]))
                    gb1[h]+=2*(1/nf)*weights2[m][h]*(-dif)#*(layer2[h]*(1-layer2[h]))
        else:
            score+=1
    y+=1
    print(score)
    if y==stopper:
        y=0
        print(yy)
        yy+=1
        if stopper==30:
            stopper=20
        else:
            stopper=30
    weights1+=grad1
    weights2+=grad2
    biases1+=gb1
    biases2+=gb2
    scal=0
    if np.max(weights1)>1 or np.min(weights1)<-1:
        scal=1
    elif np.max(biases1)>1 or np.min(biases1)<-1:
        scal=1
    if np.max(weights2)>1 or np.min(weights2)<-1:
        if scal==1:
            scal=3
        else:
            scal=2
    elif np.max(biases2)>1 or np.min(biases2)<-1:
        if scal==1:
            scal=3
        else:
            scal=2
    '''
    if np.max(weights1)>10 or np.min(weights1)<-10:
        scal=1
    elif np.max(biases1)>10 or np.min(biases1)<-10:
        scal=1
    if np.max(weights2)>10 or np.min(weights2)<-10:
        if scal==1:
            scal=3
        else:
            scal=2
    elif np.max(biases2)>10 or np.min(biases2)<-10:
        if scal==1:
            scal=3
        else:
            scal=2
    '''
    if scal!=0:
        print('scaled')
        if scal==1:
            weights1*=0.01
            biases1*=0.01
            #biases2*=0.01
            #thescaler*=100
        elif scal==2:
            weights2*=0.01
            biases2*=0.01
        elif scal==3:
            weights1*=0.01
            weights2*=0.01
            #thescaler*=100
            biases1*=0.01
            biases2*=0.0001
            #biases2*=0.01
    if score==len(feats2):
        weights1=pd.DataFrame(weights1)
        weights2=pd.DataFrame(weights2)
        weights1.to_csv('weights_3_1.csv')
        weights2.to_csv('weights_3_2.csv')
        biases1=pd.DataFrame(biases1)
        biases2=pd.DataFrame(biases2)
        biases1.to_csv('biases_3_1.csv')
        biases2.to_csv('biases_3_2.csv')
        with open("thescore.txt", "wb") as fp:   #Pickling
            pickle.dump(score, fp)
        #with open("thescaler.txt", "wb") as fp:   #Pickling
            #pickle.dump(thescaler, fp)
        break

