#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#reading the application data
appl=pd.read_csv('application_data.csv')


# In[3]:


#checking the head of data
appl.head()


# In[4]:


appl.shape


# In[5]:


print(list(appl.columns))


# In[6]:


emptycol=appl.isnull().sum()
emptycol[emptycol>0.3*len(appl)]


# In[7]:


appl.OCCUPATION_TYPE.value_counts(normalize=True)


# In[8]:


emptycol=emptycol[emptycol>0.3*len(appl)]
emptycol=list(emptycol.index)


# In[9]:


emptycol.remove('OCCUPATION_TYPE')


# In[10]:


emptycol


# In[11]:


appl=appl.drop(emptycol,axis=1)


# In[12]:


appl.head()


# In[13]:


list(appl.columns)


# In[14]:


flagdoc=[x for x in list(appl.columns) if "FLAG_DOCUMENT" in x]


# In[15]:


appl=appl.drop(flagdoc,axis=1)


# In[16]:


appl.head()


# In[17]:


appl.columns


# In[18]:


unwanted=['EXT_SOURCE_2', 'EXT_SOURCE_3',
       'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR']


# In[19]:


appl=appl.drop(unwanted,axis=1)


# In[20]:


appl.head()


# In[21]:


appl.columns


# In[22]:


appl.REGION_POPULATION_RELATIVE.describe()


# In[23]:


appl.DAYS_BIRTH.value_counts()


# In[24]:


appl.DAYS_BIRTH


# In[25]:


appl.head()


# In[26]:


9461/365


# In[27]:


flagvar=list(appl.columns)
flagvar=[x for x in flagvar if 'FLAG' in x]


# In[28]:


flagvar


# In[29]:


flagvar.remove('FLAG_OWN_REALTY')


# In[30]:


appl=appl.drop(flagvar, axis=1)


# In[31]:


appl.head()


# In[32]:


appl.columns


# In[33]:


unwanted=['NAME_TYPE_SUITE','REGION_POPULATION_RELATIVE','DAYS_EMPLOYED','REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY']


# In[34]:


len(unwanted)


# In[35]:


appl=appl.drop(unwanted,axis=1)


# In[36]:


appl.head()


# In[37]:


unwanted=['DAYS_REGISTRATION','DAYS_ID_PUBLISH']


# In[38]:


appl=appl.drop(unwanted,axis=1)


# In[39]:


appl.head()


# In[40]:


appl.isnull().sum()


# In[41]:


appl.AMT_ANNUITY.describe()


# In[42]:


appl[appl.AMT_ANNUITY.isnull()]


# In[43]:


appl.corr()


# In[44]:


appl=appl[-appl.AMT_ANNUITY.isnull()]


# In[45]:


appl.isnull().sum()


# In[46]:


appl[['AMT_CREDIT','AMT_GOODS_PRICE']]


# In[47]:


appl[appl.AMT_GOODS_PRICE==appl.AMT_CREDIT]


# In[48]:


len(appl[appl.AMT_GOODS_PRICE==appl.AMT_CREDIT])


# In[49]:


len(appl[appl.AMT_GOODS_PRICE>appl.AMT_CREDIT])


# In[50]:


len(appl[appl.AMT_GOODS_PRICE<appl.AMT_CREDIT])


# In[51]:


appl=appl[-appl.AMT_GOODS_PRICE.isnull()]


# In[52]:


appl.head()


# In[53]:


appl.shape


# In[54]:


appl.isnull().sum()


# In[55]:


appl.OCCUPATION_TYPE.value_counts()


# In[56]:


appl.head()


# In[57]:


appl.shape


# In[58]:


appl[appl.OCCUPATION_TYPE.isnull()]


# In[59]:


appl.OCCUPATION_TYPE.fillna('Not Known', inplace=True)


# In[60]:


appl.isnull().sum()


# In[61]:


appl.ORGANIZATION_TYPE.value_counts()


# In[62]:


appl.info()


# In[63]:


appl.DAYS_BIRTH.value_counts()


# In[64]:


appl.DAYS_BIRTH=appl.DAYS_BIRTH.apply(lambda x: abs(x//365) )


# In[ ]:





# In[65]:


def agegroup(x):
    if x>90:
        return '90 above'
    elif 80<x<=90:
        return '81-90'
    elif 70<x<=80:
        return '71-80'
    elif 60<x<=70:
        return '61-70'
    elif 51<x<=60:
        return '51-60'
    elif 40<x<=50:
        return '41-50'
    elif 30<x<=40:
        return '31-40'
    elif 20<=x<=30:
        return '20-30'
    else:
        return 'Teenagers'
    
    


# In[66]:


appl.DAYS_BIRTH=appl.DAYS_BIRTH.apply(agegroup)


# In[67]:


appl.DAYS_BIRTH.value_counts()


# In[68]:


appl.head()


# In[69]:


appl.NAME_FAMILY_STATUS.value_counts()


# In[70]:


appl.head()


# In[71]:


appl.ORGANIZATION_TYPE.value_counts()


# In[72]:


appl.NAME_CONTRACT_TYPE.value_counts()


# In[73]:


appl=appl.drop(appl.loc[appl.ORGANIZATION_TYPE=="XNA"].index)


# In[74]:


appl.shape


# In[75]:


appl[appl.CODE_GENDER=='XNA']


# In[76]:


appl.loc[appl['CODE_GENDER']=='XNA','CODE_GENDER']='F'
appl['CODE_GENDER'].value_counts()


# In[77]:


appl.head()


# In[78]:


# Creating bins for income amount

bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
slot = ['0-25000', '25000-50000','50000-75000','75000,100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']

appl['AMT_INCOME_RANGE']=pd.cut(appl['AMT_INCOME_TOTAL'],bins,labels=slot)


# In[79]:


# Creating bins for Credit amount

bins = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]
slots = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',
        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',
        '800000-850000','850000-900000','900000 and above']

appl['AMT_CREDIT_RANGE']=pd.cut(appl['AMT_CREDIT'],bins=bins,labels=slots)


# In[80]:


appl.head()


# In[81]:


appl_target0=appl[appl.TARGET==0]
appl_target1=appl[appl.TARGET==1]


# In[82]:


# Count plotting in logarithmic scale

def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    temp = pd.Series(data = hue)
   
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='plasma') 
        
    plt.show()


# ## Analyasis

# In[83]:


# PLotting for income range

uniplot(appl_target0,col='AMT_INCOME_RANGE',title='Distribution of income range',hue='CODE_GENDER')


# Points to be concluded from the above graph.<br>
# 
# ---Male counts are higher than female.<br>
# ---Income range from 100000 to 200000 is having more number of credits.<br>
# ---This graph show that males are more than female in having credits for that range.<br>
# ---Very less count for income range 400000 and above.

# In[84]:


# Plotting for Income type

uniplot(appl_target0,col='NAME_INCOME_TYPE',title='Distribution of Income type',hue='CODE_GENDER')


# Points to be concluded from the above graph.<br>
# 
# ---For income type ‘working’, ’commercial associate’, and ‘State Servant’ the number of credits are higher than other i.e. ‘Maternity leave.<br>
# ---For this Females are having more number of credits than male.<br>
# ---Less number of credits for income type ‘Maternity leave’.<br>
# ---For type 1: There is no income type for ‘student’ , ’pensioner’ and ‘Businessman’ which means they don’t do any late payments.<br>

# In[182]:


#plotting for agerange

plt.figure(figsize=[19,10])
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

plt.title('Distribution between Age Group')
plt.xticks(rotation=45)

sns.countplot(data=appl_target0,x='DAYS_BIRTH',order=appl_target0.DAYS_BIRTH.value_counts().index,hue='CODE_GENDER',palette='twilight')
plt.show()


# Points to be concluded from the above graph.<br>
# 
# --Age between 31-50 has higher no of credit for both male and female.<br>
# --Also age group of 20-30 & 51-60 are also taking credit
# 

# In[86]:




plt.figure(figsize=[19,10])
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

plt.title('Distribution between OCCUPATION_TYPE')
plt.xticks(rotation=45)

sns.countplot(data=appl_target0,x='OCCUPATION_TYPE',order=appl_target0.OCCUPATION_TYPE.value_counts().index,palette='rainbow')
plt.show()


# From the above graph , we had concluded that laboures,sales staff, core staff, Mangers and drivers are higher number of count as compare to customer of any other occupation

# In[87]:



plt.figure(figsize=[19,10])
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

plt.title('Distribution between OCCUPATION_TYPE wrt GENDER')
plt.xticks(rotation=45)

sns.countplot(data=appl_target0,x='OCCUPATION_TYPE',order=appl_target0.OCCUPATION_TYPE.value_counts().index,hue='CODE_GENDER',palette='rainbow')
plt.show()


# In[88]:


plt.figure(figsize=[19,36])
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

plt.title('Distribution between ORGANIZATION_TYPE')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.xscale('log')

sns.countplot(data=appl_target0,y='ORGANIZATION_TYPE',order=appl_target0.ORGANIZATION_TYPE.value_counts().index,palette='deep')
plt.show()


# Points to be concluded from the above graph.<br>
# 
# --Clients which have applied for credits are from most of the organization type ‘Business entity Type 3’ , ‘Self employed’, ‘Other’ , ‘Medicine’ and ‘Government’.<br>
# --Less clients are from Industry type 8,type 6, type 10, religion and trade type 5, type 4.

# In[89]:


uniplot(appl_target0,col='NAME_CONTRACT_TYPE',hue='CODE_GENDER',title='Distribution of NAME_CONTRACT_TYPE')


# Points to be concluded from the above graph.<br>
# 
# For contract type ‘cash loans’ is having higher number of credits than ‘Revolving loans’ contract type.<br>
# For this also Female is leading for applying credits.

# In[90]:


appl_target0.head()


# In[183]:


plt.figure(figsize=[19,10])
sns.set(style="dark", context="talk")
plt.style.use("dark_background")

plt.title('Distribution between EDUCATION_TYPE wrt GENDER')
plt.xticks(rotation=45)

sns.countplot(data=appl_target0,y='NAME_EDUCATION_TYPE',order=appl_target0.NAME_EDUCATION_TYPE.value_counts().index,hue='CODE_GENDER',palette='rainbow')
plt.show()


# From above graph we had cocluded that client posessing higher education and secondry/secondry special taking higher credit

# In[92]:


uniplot(appl_target0,col='NAME_FAMILY_STATUS',hue='CODE_GENDER',title='NAME_FAMILY_STATUS')


# Number of married people is higher as compare to client having any other status

# In[93]:


# PLotting for income range

uniplot(appl_target1,col='AMT_INCOME_RANGE',title='Distribution of income range',hue='CODE_GENDER')


# Points to be concluded from the above graph.<br>
# 
# ---Male counts are higher than female.<br>
# ---Income range from 100000 to 200000 is having more number of credits.<br>
# ---This graph show that males are more than female in having credits for that range.<br>
# ---Very less count for income range 400000 and above.

# In[94]:


uniplot(appl_target1,col='NAME_INCOME_TYPE',title='Distribution of Income type',hue='CODE_GENDER')


# Points to be concluded from the above graph.<br>
# 
# For income type ‘working’, ’commercial associate’, and ‘State Servant’ the number of credits are higher than other i.e. ‘Maternity leave.<br>
# For this Females are having more number of credits than male.<br>
# Less number of credits for income type ‘Maternity leave’.<br>
# For type 1: There is no income type for ‘student’  and ‘Businessman’ which means they don’t do any late payments.<br>

# In[185]:


#plotting for agerange


plt.figure(figsize=[19,10])
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

plt.title('Distribution between Age Group')
plt.xticks(rotation=45)

sns.countplot(data=appl_target1,x='DAYS_BIRTH',order=appl_target0.DAYS_BIRTH.value_counts().index,hue='CODE_GENDER',palette='twilight')
plt.show()


# In[96]:


plt.figure(figsize=[19,10])
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

plt.title('Distribution between OCCUPATION_TYPE')
plt.xticks(rotation=45)

sns.countplot(data=appl_target1,x='OCCUPATION_TYPE',order=appl_target0.OCCUPATION_TYPE.value_counts().index,palette='rainbow')
plt.show()


# In[97]:


plt.figure(figsize=[19,10])
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

plt.title('Distribution between OCCUPATION_TYPE wrt GENDER')
plt.xticks(rotation=45)

sns.countplot(data=appl_target1,x='OCCUPATION_TYPE',order=appl_target0.OCCUPATION_TYPE.value_counts().index,hue='CODE_GENDER',palette='rainbow')
plt.show()


# In[98]:


plt.figure(figsize=[19,36])
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

plt.title('Distribution between ORGANIZATION_TYPE')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.xscale('log')

sns.countplot(data=appl_target1,y='ORGANIZATION_TYPE',order=appl_target0.ORGANIZATION_TYPE.value_counts().index,palette='deep')
plt.show()


# Points to be concluded from the above graph.<br>
# 
# Clients which have applied for credits are from most of the organization type ‘Business entity Type 3’ , ‘Self employed’ , ‘Other’ , ‘Medicine’ and ‘Government’.<br>
# Less clients are from Industry type 8,type 6, type 10, religion and trade type 5, type 4.<br>
# Same as type 0 in distribution of organization type.<br>

# In[99]:


uniplot(appl_target1,col='NAME_CONTRACT_TYPE',hue='CODE_GENDER',title='Distribution of NAME_CONTRACT_TYPE')


# Points to be concluded from the above graph.<br>
# 
# For contract type ‘cash loans’ is having higher number of credits than ‘Revolving loans’ contract type.<br>
# For this also Female is leading for applying credits.<br>
# For type 1 : there is only Female Revolving loans.<br>

# In[100]:


uniplot(appl_target1,col='NAME_FAMILY_STATUS',hue='CODE_GENDER',title='NAME_FAMILY_STATUS')


# In[113]:


appl_target0_corr=appl_target0.iloc[0:,2:]
appl_target1_corr=appl_target1.iloc[0:,2:]
appl_target0_cor=appl_target0_corr.corr()
appl_target1_corr=appl_target1_corr.corr()


# In[114]:


plt.figure(figsize=[10,12])
plt.title('correlation of target 0(non defaulter)')
plt.xticks(rotation=45)
sns.heatmap(appl_target0_cor,annot=True)
plt.show()


# As we can see from above correlation heatmap, There are number of observation we can point out,<br>
# 
# 
# Credit amount is inversely proportional to the number of children client have, means Credit amount is higher for less children count client have and vice-versa.<br>
# Income amount is inversely proportional to the number of children client have, means more income for less children client have and vice-versa.<br>
# 

# In[186]:


plt.figure(figsize=[10,12])
plt.title('correlation of target 1(defaulter)')
plt.xticks(rotation=45)
sns.heatmap(appl_target0_cor,annot=True)
plt.show()


# This heat map for Target 1 is also having quite a same observation just like Target 0.

# In[132]:


sns.boxplot(y='AMT_INCOME_TOTAL',data=appl_target0)
    


# Few points can be concluded from the graph above.<br>
# 
# Some outliers are noticed in income amount.<br>
# The third quartiles is very slim for income amount.

# In[128]:


sns.boxplot(y='AMT_CREDIT',data=appl_target0)


# Few points can be concluded from the graph above.<br>
# 
# Some outliers are noticed in credit amount.<br>
# The first quartile is bigger than third quartile for credit amount which means most of the credits of clients are present in the first quartile.

# In[133]:


sns.boxplot(y='AMT_ANNUITY',data=appl_target0)


# Few points can be concluded from the graph above.<br>
# 
# Some outliers are noticed in annuity amount.<br>
# The first quartile is bigger than third quartile for annuity amount which means most of the annuity clients are from first quartile.

# In[134]:


sns.boxplot(y='AMT_INCOME_TOTAL',data=appl_target1)


# Few points can be concluded from the graph above.<br>
# 
# Some outliers are noticed in income amount.<br>
# The third quartiles is very slim for income amount.<br1>
# Most of the clients of income are present in first quartile

# In[135]:


sns.boxplot(y='AMT_CREDIT',data=appl_target1)


# Few points can be concluded from the graph above.<br>
# 
# Some outliers are noticed in credit amount.<br>
# The first quartile is bigger than third quartile for credit amount which means most of the credits of clients are present in the first quartile.
# 

# In[187]:


sns.boxplot(y='AMT_ANNUITY',data=appl_target1)


# Few points can be concluded from the graph above.<br>
# 
# Some outliers are noticed in annuity amount.<br>
# The first quartile is bigger than third quartile for annuity amount which means most of the annuity clients are from first quartile.

# In[139]:


plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
sns.boxplot(data =appl_target0, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Credit amount vs Education Status')
plt.show()


# From the above box plot we can conclude that Family status of 'civil marriage', 'marriage' and 'separated' of Academic degree education are having higher number of credits than others. Also, higher education of family status of 'marriage', 'single' and 'civil marriage' are having more outliers. Civil marriage for Academic degree is having most of the credits in the third quartile.

# In[140]:


# Box plotting for Income amount in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
plt.yscale('log')
sns.boxplot(data =appl_target0, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Income amount vs Education Status')
plt.show()


# From above boxplot for Education type 'Higher education' the income amount is mostly equal with family status. It does contain many outliers. Less outlier are having for Academic degree but there income amount is little higher that Higher education. Lower secondary of civil marriage family status are have less income amount than others.

# ##### For Target 1

# In[141]:


# Box plotting for credit amount

plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
sns.boxplot(data =appl_target1, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Credit Amount vs Education Status')
plt.show()


# In[142]:


# Box plotting for Income amount in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
plt.yscale('log')
sns.boxplot(data =appl_target1, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Income amount vs Education Status')
plt.show()


# In[144]:


appl_target0.head()


# In[147]:


prev=pd.read_csv('previous_application.csv')


# In[148]:


prev.head()


# In[149]:


# Cleaning the missing data

# listing the null values columns having more than 30%

emptycol1=prev.isnull().sum()
emptycol1=emptycol1[emptycol1.values>(0.3*len(prev))]
len(emptycol1)


# In[150]:


# Removing those 11 columns

emptycol1 = list(emptycol1[emptycol1.values>=0.3].index)
prev.drop(labels=emptycol1,axis=1,inplace=True)

prev.shape


# In[152]:


print(list(prev.columns))


# In[159]:


unwanted=['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY','NAME_PORTFOLIO','DAYS_DECISION','NAME_PAYMENT_TYPE','CHANNEL_TYPE', 'SELLERPLACE_AREA', 'NAME_SELLER_INDUSTRY', 'CNT_PAYMENT', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']


# In[161]:


prev.drop(unwanted,axis=1,inplace=True)


# In[162]:


prev.head()


# In[163]:


unwanted=['CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PRODUCT_TYPE']
prev.drop(unwanted,axis=1,inplace=True)


# In[164]:


prev.head()


# In[165]:


prev=prev.drop(prev[prev['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)
prev=prev.drop(prev[prev['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)
prev=prev.drop(prev[prev['NAME_CASH_LOAN_PURPOSE']=='XAP'].index)


# In[167]:


# Distribution of contract status in logarithmic scale

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of contract status with purposes')
ax = sns.countplot(data = prev, y= 'NAME_CASH_LOAN_PURPOSE', 
                   order=prev['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_CONTRACT_STATUS',palette='magma') 


# Points to be concluded from above plot:<br>
# 
# Most rejection of loans came from purpose 'repairs'.<br>
# For education purposes we have equal number of approves and rejection<br>
# Payign other loans and buying a new car is having significant higher rejection than approves.

# In[173]:


new_df=pd.merge(left=appl,right=prev,how='inner',on='SK_ID_CURR',suffixes='_x')


# In[174]:


new_df.head()


# In[176]:


# Distribution of contract status

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of purposes with target ')
ax = sns.countplot(data = new_df, y= 'NAME_CASH_LOAN_PURPOSE', 
                   order=new_df['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'TARGET',palette='magma') 


# Few points we can conclude from abpve plot:<br>
# 
# Loan purposes with 'Repairs' are facing more difficulites in payment on time.<br>
# There are few places where loan payment is significant higher than facing difficulties. They are 'Buying a garage', 'Business developemt', 'Buying land','Buying a new car' and 'Education'<br>
# Hence we can focus on these purposes for which the client is having for minimal payment difficulties.<br>

# In[178]:


# Box plotting for Credit amount in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
plt.yscale('log')
sns.boxplot(data =new_df, x='NAME_CASH_LOAN_PURPOSE',hue='NAME_INCOME_TYPE',y='AMT_CREDITx',orient='v')
plt.title('Prev Credit amount vs Loan Purpose')
plt.show()


# From the above we can conclude some points-<br>
# 
# The credit amount of Loan purposes like 'Buying a home','Buying a land','Buying a new car' and'Building a house' is higher.<br>
# Income type of state servants have a significant amount of credit applied<br>
# Money for third person or a Hobby is having less credits applied for.

# In[180]:


# Box plotting for Credit amount prev vs Housing type in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
sns.barplot(data =new_df, y='AMT_CREDITx',hue='TARGET',x='NAME_HOUSING_TYPE')
plt.title('Prev Credit amount vs Housing type')
plt.show()


# Here for Housing type, office appartment is having higher credit of target 0 and co-op apartment is having higher credit of target 1. So, we can conclude that bank should avoid giving loans to the housing type of co-op apartment as they are having difficulties in payment. Bank can focus mostly on housing type with parents or House\appartment or miuncipal appartment for successful payments.

# # CONCLUSION

# --Banks should focus more on contract type ‘Student’ ,’pensioner’ and ‘Businessman’ with housing ‘type other than ‘Co-op apartment’ and 'office appartment' for successful payments.<br>
# --Banks should focus less on income types maternity leave and working as they have most number of unsuccessful payments
# In loan purpose ‘Repairs’:<br>
# 
#        a. Although having higher number of rejection in loan purposes with 'Repairs' we can observe difficulties in payment.
#        b. There are few places where loan payment diffuculty is significantly high.
#        c. Bank should continue to be cautious while giving loan for this purpose.
# --Bank can focus mostly on housing type with parents , House or apartment and municipal apartment with purpuse of education, buying land, buying a garage, purchase of electronic equipment and some other purposes with target0 significantly more than target1 for successful payments.<br>
# --Banks can offer more offers to clients who are students and pensioners as they take all offers and are more likely to pay back

# * CODE_GENDER: Men are at relatively higher default rate<br>
# * NAME_FAMILY_STATUS : People who have civil marriage or who are single default a lot.<br>
# * NAME_EDUCATION_TYPE: People with Lower Secondary & Secondary education<br>
# * NAME_INCOME_TYPE: Clients who are either at Maternity leave OR Unemployed default a lot.<br>
# * REGION_RATING_CLIENT: People who live in Rating 3 has highest defaults.<br>
# * OCCUPATION_TYPE: Avoid Low-skill Laborers, Drivers and Waiters/barmen staff, Security staff, Laborers and Cooking staff as the default rate is huge.<br>
# * ORGANIZATION_TYPE: Organizations with highest percent of loans not repaid are Transport: type 3 (16%), Industry: type 13 (13.5%), Industry: type 8 (12.5%) and Restaurant (less than 12%). Self-employed people have relative high defaulting rate, and thus should be avoided to be approved for loan or provide loan with higher interest rate to mitigate the risk of defaulting.<br>
# * DAYS_BIRTH: Avoid young people who are in age group of 20-40 as they have higher probability of defaulting<br>
# * DAYS_EMPLOYED: People who have less than 5 years of employment have high default rate.<br>
# * CNT_CHILDREN & CNT_FAM_MEMBERS: Client who have children equal to or more than 9 default 100% and hence their applications are to be rejected.<br>
# * AMT_GOODS_PRICE: When the credit amount goes beyond 3M, there is an increase in defaulters.<br>

# ### The following attributes indicate that people from these category tend to default but then due to the number of people and the amount of loan, the bank could provide loan with higher interest to mitigate any default risk thus preventing business loss:
# 
# * NAME_HOUSING_TYPE: High number of loan applications are from the category of people who live in Rented apartments & living with parents and hence offering the loan would mitigate the loss if any of those default.<br>
# * AMT_CREDIT: People who get loan for 300-600k tend to default more than others and hence having higher interest specifically for this credit range would be ideal.<br>
# * AMT_INCOME: Since 90% of the applications have Income total less than 300,000 and they have high probability of defaulting, they could be offered loan with higher interest compared to other income category.<br>
# * CNT_CHILDREN & CNT_FAM_MEMBERS: Clients who have 4 to 8 children has a very high default rate and hence higher interest should be imposed on their loans.<br>
# * NAME_CASH_LOAN_PURPOSE: Loan taken for the purpose of Repairs seems to have highest default rate. A very high number applications have been rejected by bank or refused by client in previous applications as well which has purpose as repair or other. This shows that purpose repair is taken as high risk by bank and either they are rejected, or bank offers very high loan interest rate which is not feasible by the clients, thus they refuse the loan. The same approach could be followed in future as well.
# 

# ### Other suggestions:
# * 90% of the previously cancelled client have actually repayed the loan. Record the reason for cancellation which might help the bank to determine and negotiate terms with these repaying customers in future for increase business opportunity.<br>
# * 88% of the clients who were refused by bank for loan earlier have now turned into a repaying client. Hence documenting the reason for rejection could mitigate the business loss and these clients could be contacted for further loans.

# In[ ]:




