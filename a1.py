import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

plt.rc('font', **font)


df = pd.read_excel (r'C:\Users\yangy\Desktop\Year4_Academic\APM466\Assignment1\a1_data_selected.xlsx')

#Find bond0's yield curve  #semiannual compounding, exchange has clean price+ accrued interst = dirty price and equals to pv of cash flows
def bond1():
    n=1
    yd = []
    for j in range(4,14):
        y=0.01
        t =[59,58,55,54,53,52,51,48,47,46] #discount time
        #accrued interest = 182-t
        p = df.loc[0][17-j]
        coupon = df.loc[0][1]
        while abs(p + coupon * 100 * (182 - t[j-4]) / 365 - ((coupon * 100 / y * (1 - (1 / ((1 + y / 2) ** n))) + 100 / ((1 + y / 2) ** n)) * ((1 + y / 2) ** ((182 - t[j-4]) / 182.5)))) >= 0.001:
                y = y+0.00001
        yd.append(y)
    return yd


def bond2():
    n=2
    yd = []
    for j in range(4,14):
        y=0.01
        t =[59,58,55,54,53,52,51,48,47,46] #discount time
        #accrued interest = 182-t
        p = df.loc[1][17-j]
        coupon = df.loc[1][1]
        while abs(p + coupon * 100 * (182 - t[j-4]) / 365 - ((coupon * 100 / y * (1 - (1 / ((1 + y / 2) ** n))) + 100 / ((1 + y / 2) ** n)) * ((1 + y / 2) ** ((182 - t[j-4]) / 182.5)))) >= 0.001:
                y = y+0.00001
        yd.append(y)
    return yd

def bond3():
    n=3
    yd = []
    for j in range(4,14):
        y=0.01
        t =[59,58,55,54,53,52,51,48,47,46] #discount time
        #accrued interest = 182-t
        p = df.loc[2][17-j]
        coupon = df.loc[2][1]
        while abs(p + coupon * 100 * (182 - t[j-4]) / 365 - ((coupon * 100 / y * (1 - (1 / ((1 + y / 2) ** n))) + 100 / ((1 + y / 2) ** n)) * ((1 + y / 2) ** ((182 - t[j-4]) / 182.5)))) >= 0.001:
                y = y+0.00001
        yd.append(y)
    return yd

def bond4():
    n=4
    yd = []
    for j in range(4,14):
        y=0.01
        t =[59,58,55,54,53,52,51,48,47,46] #discount time
        #accrued interest = 182-t
        p = df.loc[3][17-j]
        coupon = df.loc[3][1]
        while abs(p + coupon * 100 * (182 - t[j-4]) / 365 - ((coupon * 100 / y * (1 - (1 / ((1 + y / 2) ** n))) + 100 / ((1 + y / 2) ** n)) * ((1 + y / 2) ** ((182 - t[j-4]) / 182.5)))) >= 0.001:
                y = y+0.00001
        yd.append(y)
    return yd

def bond5():
    n=5
    yd = []
    for j in range(4,14):
        y=0.01
        t =[59,58,55,54,53,52,51,48,47,46] #discount time
        #accrued interest = 182-t
        p = df.loc[4][17-j]
        coupon = df.loc[4][1]
        while abs(p + coupon * 100 * (182 - t[j-4]) / 365 - ((coupon * 100 / y * (1 - (1 / ((1 + y / 2) ** n))) + 100 / ((1 + y / 2) ** n)) * ((1 + y / 2) ** ((182 - t[j-4]) / 182.5)))) >= 0.001:
                y = y+0.00001
        yd.append(y)
    return yd

# check if it is correct
def bond6():
    n=5
    yd = []
    for j in range(4,14):
        y=0.01
        t =[151,150,147,146,145,144,143,140,139,138] #discount time
        #accrued interest = 182-t
        p = df.loc[5][17-j]
        coupon = df.loc[5][1]
        while abs(p + coupon * 100 * (182 - t[j-4]) / 365 - ((coupon * 100 / y * (1 - (1 / ((1 + y / 2) ** n))) + 100 / ((1 + y / 2) ** n)) * ((1 + y / 2) ** ((182 - t[j-4]) / 182.5)))) >= 0.005:
                y = y+0.00001

        yd.append(y)
    return yd


def bond7():
    n=7
    yd = []
    for j in range(4,14):
        y=0.01
        t =[59,58,55,54,53,52,51,48,47,46] #discount time
        #accrued interest = 182-t
        p = df.loc[6][17-j]
        coupon = df.loc[6][1]
        while abs(p + coupon * 100 * (182 - t[j-4]) / 365 - ((coupon * 100 / y * (1 - (1 / ((1 + y / 2) ** n))) + 100 / ((1 + y / 2) ** n)) * ((1 + y / 2) ** ((182 - t[j-4]) / 182.5)))) >= 0.005:
                y = y+0.00001
        yd.append(y)
    return yd

def bond8():
    n=7
    yd = []
    for j in range(4,14):
        y=0.01
        t =[151,150,147,146,145,144,143,140,139,138] #discount time
        #accrued interest = 182-t
        p = df.loc[7][17-j]
        coupon = df.loc[7][1]
        while abs(p + coupon * 100 * (182 - t[j-4]) / 365 - ((coupon * 100 / y * (1 - (1 / ((1 + y / 2) ** n))) + 100 / ((1 + y / 2) ** n)) * ((1 + y / 2) ** ((182 - t[j-4]) / 182.5)))) >= 0.005:
                y = y+0.00001
        yd.append(y)
    return yd

def bond9():
    n=9
    yd = []
    for j in range(4,14):
        y=0.01
        t =[59,58,55,54,53,52,51,48,47,46] #discount time
        #accrued interest = 182-t
        p = df.loc[8][17-j]
        coupon = df.loc[8][1]
        while abs(p + coupon * 100 * (182 - t[j-4]) / 365 - ((coupon * 100 / y * (1 - (1 / ((1 + y / 2) ** n))) + 100 / ((1 + y / 2) ** n)) * ((1 + y / 2) ** ((182 - t[j-4]) / 182.5)))) >= 0.005:
                y = y+0.00001
        yd.append(y)
    return yd

def bond10():
    n=10
    yd = []
    for j in range(4,14):
        y=0.01
        t =[59,58,55,54,53,52,51,48,47,46] #discount time
        #accrued interest = 182-t
        p = df.loc[9][17-j]
        coupon = df.loc[9][1]
        while abs(p + coupon * 100 * (182 - t[j-4]) / 365 - ((coupon * 100 / y * (1 - (1 / ((1 + y / 2) ** n))) + 100 / ((1 + y / 2) ** n)) * ((1 + y / 2) ** ((182 - t[j-4]) / 182.5)))) >= 0.005:
                y = y+0.00001
        yd.append(y)
    return yd


def bond11():
    n=11
    yd = []
    for j in range(4,14):
        y=0.01
        t =[59,58,55,54,53,52,51,48,47,46] #discount time
        #accrued interest = 182-t
        p = df.loc[10][17-j]
        coupon = df.loc[10][1]
        while abs(p + coupon * 100 * (182 - t[j-4]) / 365 - ((coupon * 100 / y * (1 - (1 / ((1 + y / 2) ** n))) + 100 / ((1 + y / 2) ** n)) * ((1 + y / 2) ** ((182 - t[j-4]) / 182.5)))) >= 0.005:
                y = y+0.00001
        yd.append(y)
    return yd



#

bond1 = bond1()
bond2 = bond2()
bond3 = bond3()
bond4 = bond4()
bond5 = bond5()
bond6 = bond6()
bond7 = bond7()
bond8 = bond8()
bond9 = bond9()
bond10 = bond10()
bond11 = bond11()

ytm = []

for i in range(10):
    ytm.append([bond1[i], bond2[i],bond3[i],bond4[i],bond5[i],bond6[i],bond7[i],bond8[i],bond9[i],bond10[i],bond11[i]])

t =[59,58,55,54,53,52,51,48,47,46]
interpolate_ytm = []
for i in range(10):
    y_1 = ytm[i][1] + (ytm[i][2]-ytm[1][1])*(213-t[i])/181
    y_2 = ytm[i][3] + (ytm[i][4]-ytm[i][3])*(182-t[i])/181
    y_3 = ytm[i][5] + (ytm[i][6]-ytm[i][5])*(274-t[i])/273
    y_4 = ytm[i][7] + (ytm[i][8]-ytm[i][7])*(274-t[i])/274
    y_5 = ytm[i][9] + (ytm[i][10]-ytm[i][9])*(182-t[i])/181
    interpolate_ytm.append([y_1,y_2,y_3,y_4,y_5])

years = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
ytm_plot = []
for i in range(10):
    ytm_plot.append([bond2[i],bond3[i],bond4[i],bond5[i],bond6[i],bond7[i],bond8[i],bond9[i],bond10[i],bond11[i]])
for j in range(10):
    plt.plot(years, ytm_plot[j], label='Jan' + str(31 - t[j]+30))

plt.xlabel('Year')
plt.ylabel('Yield to Maturity')
plt.title('Yield Curve')
plt.legend()
plt.show()



#Spot Curve



def sr0(t,p):
    y = ((100.75/(p+1.5*(182-6)/365))**(182.5/t)-1)*2
    return y


def sr1(sr0,t,p):
    pay1 = 0.375/(1+sr0/2)**(t/182.5)
    BAI = p + 0.75* (182.5 - t) / 365
    y = (100.375/(BAI - pay1)) ** (182.5/(t+182.5)) -1
    y = y*2
    return y


def sr2(sr0,sr1,t,p):
    BAI = p + 0.75*(182.5-t)/365
    pay1 = 0.375/(1+sr0/2)**(t/182.5)
    pay2 = 0.375/(1+sr1/2)**(t/182.5+1)
    y = (100.375/(BAI-pay1-pay2))**(182.5/(t+365))-1
    y = y*2
    return y

def z1(z_0,p,t):
    price = p + 0.75*(31-t+121)/365
    pmt1 = 0.375/(1+0.5*z_0)**((t+29)/182.5)
    semi = (100.375/(price-pmt1))**(182.5/(t+213))-1
    y = semi*2
    return y

def sr3(sr0, sr1 ,sr2 ,t,p):
    BAI = p + 0.75*(182.5-t)/365
    pay1 = 0.375/(1+sr0/2)**(t/182.5)
    pay2 = 0.375/(1+sr1/2)**(t/182.5+1)
    pay3 = 0.375/(1+sr2/2)**(t/182.5+2)
    y = (100.375/(BAI-pay1-pay2-pay3))**(182.5/(t+547.5))-1
    y = y*2
    return y

def sr4(sr0,sr1,sr2,sr3,t,p):
    BAI = p + 0.5*(182.5-t)/365
    pay1 = 0.25/(1+sr0/2)**(t/182.5)
    pay2 = 0.25/(1+sr1/2)**(t/182.5+1)
    pay3 = 0.25/(1+sr2/2)**(t/182.5+2)
    pay4 = 0.25/(1+0.5*sr3)**(t/182.5+3)
    y = (100.25/(BAI-pay1-pay2-pay3-pay4))**(182.5/(t+730))-1
    y = y*2
    return y

def sr4_diff(sr0,sr1,sr2,sr3,sr4,t,p):
    sr_diff = []
    sr =[sr0, sr1, sr2, sr3, sr4]
    payments = 0
    for i in range(4):
        sr_diff.append(sr[i]+(sr[i+1]-sr[i])*92/183)

    BAI = 2.75*(t-27)/365 + p
    pay1 = 1.375/(1+sr_diff[0]/2)**((t+92)/182.5)
    pay2 = 1.375/(1+sr_diff[1]/2)**((t+275)/182.5)
    pay3 = 1.375/(1+sr_diff[2]/2)**((t+457)/182.5)
    pay4 = 1.375/(1+sr_diff[3]/2)**((t+640)/182.5)
    y = (101.375/(BAI-pay1-pay2-pay3-pay4))**(182.5/(t+823))-1
    y = y*2
    return y


def sr6(sr0,sr1,sr2,sr3,sr4,sr4_diff,t,p):
    BAI = p + 1.75*(182.5-t)/365
    sr = [sr0,sr1,sr2,sr3,sr4]
    payments = 0
    for i in range (5):
        payments = payments + 0.875/(1+sr[i]/2)**(t/182.5+i)
    sr6 = 0.001
    while 0.875/(1+0.5*(sr4_diff+(sr6-sr4_diff)*92/273))**((t+914)/182.5) + 100.875/(1+sr6/2)**((t+1095)/182.5) - (BAI-payments) > 0.0001:
            sr6 = sr6 + 0.0001
    sr5 = (sr4_diff + (sr6 - sr4_diff)*92/273)
    return  [sr5,sr6]


def sr6_diff(sr0,sr1,sr2,sr3,sr4,sr5,sr6,sr4_diff,t,p):
    sr = [sr0,sr1,sr2,sr3,sr4,sr5,sr6]
    sr_diff = []

    for i in range(4):
        int = sr[i]+ (sr[i+1]-sr[i]) * 92/182.5
        sr_diff.append(int)
    sr5_diff = sr5 + (sr6 - sr5) * 91 / 181
    sr_diff.append(sr4_diff)
    sr_diff.append(sr5_diff)

    BAI = 1.5*(t-27)/365 + p
    pay1 = 0.75/(1+sr_diff[0]/2)**((t+92)/182.5)
    pay2 = 0.75/(1+sr_diff[1]/2)**((t+275)/182.5)
    pay3 = 0.75/(1+sr_diff[2]/2)**((t+457)/182.5)
    pay4 = 0.75/(1+sr_diff[3]/2)**((t+640)/182.5)
    pay5 = 0.75/(1+sr_diff[4]/2)**((t+823)/182.5)
    pay6 = 0.75/(1+sr_diff[5]/2)**((t+1007)/182.5)
    y = (100.75/(BAI-pay1-pay2-pay3-pay4-pay5-pay6))**(182.5/(t+1188))-1
    y = y*2
    return y

def sr8(sr0,sr1,sr2,sr3,sr4,sr5,sr6,sr6_diff,t,p):
    BAL = p + 2.25*(182.5-t)/365
    payments = 0
    sr = [sr0,sr1,sr2,sr3,sr4,sr5,sr6]
    for i in range(7):
        payments = payments + 1.125/(1+sr[i]/2)**((t/182.5)+i)
    sr8= 0.01
    while 1.125/(1+0.5*(sr6_diff+(sr8-sr6_diff)*92/273))**((t+1308)/182.5) + 101.125/(1+0.5*sr8)**((t+1489)/182.5) - (BAL-payments) > 0.0001:
        sr8 += 0.0001
    sr7 = sr6_diff + (sr8-sr6_diff)*92/373
    return [sr7,sr8]
#
def sr9(sr0,sr1,sr2,sr3,sr4,sr5,sr6,sr7,sr8,t,p):
    payments = 0
    sr = [sr0, sr1, sr2, sr3, sr4, sr5, sr6, sr7, sr8]
    for i in range(9):
        payments = payments + 0.75 / (1 + sr[i] / 2) ** ((t / 182.5) + i)
    BAL = p + 1.25*(182.5- t)/365
    y = (100.75/(BAL- payments))**(182.5/(t+1644))-1
    y = y*2
    return y


def sr10(sr0,sr1,sr2,sr3,sr4,sr5,sr6,sr7,sr8,sr9,t,p):
    BAL = p + 1.5*(182.5-t)/365
    sr = [sr0, sr1, sr2, sr3, sr4, sr5, sr6, sr7, sr8,sr9]
    payments = 0
    for i in range(10):
        payments = payments + 0.625/(1 + sr[i] / 2) ** ((t / 182.5) + i)
    y = (100.625/(BAL-payments))**(182.5/(t+1816))-1
    y = y*2
    return y


spot_results = []
t =[59,58,55,54,53,52,51,48,47,46]
interpolate_results = []
fr = [] #forward rate
for i in range (10):
    r_diff = []
    results = []
    results.append(sr0(t[i], df.loc[0][13-i]))
    results.append(sr1(results[0], t[i], df.loc[1][13-i]))
    results.append(sr2(results[0], results[1], t[i], df.loc[2][13-i]))
    results.append(sr3(results[0], results[1] ,results[2], t[i], df.loc[3][13-i]))
    results.append(sr4(results[0], results[1] ,results[2], results[3], t[i], df.loc[4][13-i]))
    r_diff.append(sr4_diff(results[0], results[1] ,results[2], results[3],results[4], t[i], df.loc[5][13-i]))
    temp1 = sr6(results[0], results[1] ,results[2], results[3],results[4], r_diff[0], t[i], df.loc[6][13-i])
    results.append(temp1[0])
    results.append(temp1[1])
    temp1 = []
    r_diff.append(sr6_diff(results[0], results[1] ,results[2], results[3],results[4],results[5], results[6], r_diff[0], t[i], df.loc[7][13-i]))
    temp2 = sr8(results[0], results[1] ,results[2], results[3],results[4],results[5], results[6],r_diff[1],t[i],df.loc[8][13-i])
    results.append(temp2[0])
    results.append(temp2[1])
    temp2 = []
    results.append(sr9(results[0], results[1] ,results[2], results[3],results[4],results[5], results[6],results[7],results[8], t[i], df.loc[9][13-i]))
    results.append(sr10(results[0], results[1] ,results[2], results[3],results[4],results[5], results[6],results[7],results[8], results[9],t[i],df.loc[10][13-i]))
    results.remove(results[0])
    spot_results.append(results)
    interpolate_spot = []

    for k in range(0,10,2):
        interpolate_spot.append(results[k] + (results[k+1]-results[k]) * ((182.5 - t[k]) / 182.5))
    interpolate_results.append(interpolate_spot)

    #find forward rate
    fd1 = (((1+interpolate_spot[1])**2)/((1+interpolate_spot[0]))-1)
    fd2 = ((((1+interpolate_spot[2])**3)/((1+interpolate_spot[0])))**(1/2)-1)
    fd3 = ((((1+interpolate_spot[3])**4)/((1+interpolate_spot[0])))**(1/3)-1)
    fd4 = ((((1+interpolate_spot[4])**5)/((1+interpolate_spot[0])))**(1/4)-1)

    fr.append([fd1,fd2,fd3,fd4])
    interpolate_spot = []


# for i in range(10):
#     years = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
#     plt.plot(years, spot_results[i], label = 'Jan'+str(31-t[i]+30))
#
# plt.xlabel('Year')
# plt.ylabel('Spot Rate')
# plt.title('Spot Curve')
#
#
# plt.legend()
# plt.show()
#
# for i in range(10):
#     years = [1,2,3,4,5]
#     plt.plot(years, interpolate_results[i], label = 'Jan'+str(31-t[i]+30))
# plt.xlabel('Year')
# plt.ylabel('Spot Rate')
# plt.title('Spot Curve')
# plt.legend()
# plt.show()

# #
# for i in range(10):
#     plt.plot(['1yr-1yr','1yr-2yr','1yr-3yr','1yr-4yr'],fr[i], label = 'Jan'+str(31-t[i]+30))
#
# plt.xlabel('Year')
# plt.ylabel('Forward Rate')
# plt.title('Forward Curve')
# plt.legend()
# plt.show()



ytm_r = []
for i in range(5):
    temp = []
    for j in range(9):
        temp.append(math.log(interpolate_ytm[j+1][i]/interpolate_ytm[j][i]))
    ytm_r.append(temp)

forward_r = []
for i in range(4):
    temp = []
    for j in range(9):
        temp.append(math.log(fr[j+1][i]/fr[j][i]))
    forward_r.append(temp)

covmatrix_ytm_r = np.cov(ytm_r)
covmatrix_f_r = np.cov(forward_r)

#Forward

e_val_ytm, e_vec_ytm = np.linalg.eig(covmatrix_ytm_r)
e_val_f, e_vec_f = np.linalg.eig(covmatrix_f_r)

print("Eigen Values:")
print(e_val_f)
print("Eigen Vectors:")
print(e_vec_f)

