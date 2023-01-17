import numpy as np

cf_com=np.zeros((2,2), dtype=int)
cf_curr=[[7, 8, 9], [1, 2, 3], [3, 2, 1]]
cf_com2 = np.zeros((2, 2), dtype=int)
for i in range(len(cf_curr)):
    cf_com[1][1]+=cf_curr[i][i]
    for j in range(len(cf_curr)):
        if i!=j:
            cf_com[0][1]+=cf_curr[j][i]
            cf_com[1][0]+=cf_curr[i][j]

    for p in range(len(cf_curr)):
        for q in range(len(cf_curr)):
            if( p!=i and q!=i):
                cf_com[0][0]+=cf_curr[p][q]
print(cf_com)

for i in range(len(cf_curr)):
    for j in range(len(cf_curr)):
        if i==j and i==0:
            cf_com2[0][0] += cf_curr[0][0]
        elif i==j and i!=0:
            cf_com2[1][1] += cf_curr[i][i]
        elif i==0:
            cf_com2[0][1] += cf_curr[0][j]
        elif j==0:
            cf_com2[1][0] += cf_curr[i][0]
        else:
            cf_com2[1][1] += cf_curr[i][j]

print(cf_com2)

