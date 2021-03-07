import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve , auc
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_pickle('/Users/Adamyae/Downloads/dataset.pkl')

#change columns names
df.columns = ['id', 'dates', 'transaction_amount', 'days_before_request', 'loan_amount', 'loan_date', 'isDefault']
#extract_id
df['id'] = df.iloc[:,0].apply(lambda x : x[1])
#extract_dates
df['dates'] = df.iloc[:,1].apply(lambda x : x[1])
#extract_transaction_amount
df['transaction_amount'] = df.iloc[:,2].apply(lambda x : x[1])
#extract_days
df['days_before_request'] = df.iloc[:,3].apply(lambda x : x[1])
#extract_loan
df['loan_amount'] = df.iloc[:,4].apply(lambda x : x[1])
#extract_date
df['loan_date'] = df.iloc[:,5].apply(lambda x : x[1])


df['transaction_amount'] = df.iloc[:,2].apply(lambda x : np.concatenate(x, axis=0))
df['days_before_request'] = df.iloc[:,3].apply(lambda x : np.concatenate(x, axis=0))

#feature_created
df['max_amount'] = df.iloc[:,2].apply(lambda x : np.max(x))
df['min_amount'] = df.iloc[:,2].apply(lambda x : np.min(x))
df['median_amount'] = df.iloc[:,2].apply(lambda x : np.median(x))
df['mean_amount'] = df.iloc[:,2].apply(lambda x : np.mean(x))

df['max_days'] = df.iloc[:,3].apply(lambda x : np.max(x))
df['min_days'] =  df.iloc[:,3].apply(lambda x : np.min(x))
df['median_days'] = df.iloc[:,3].apply(lambda x : np.median(x))
df['mean_days'] = df.iloc[:,3].apply(lambda x : np.mean(x))

df['loan_date'] = df.loc[:,'loan_date'].apply(lambda x : pd.to_datetime(x))
#reindexing and dropping dates(insignificant feature)

df = df.drop('dates', axis=1)
df.set_index('id', inplace=True)

#getting total no of debit transaction
def num_debit(array):
    n = len(array)
   
    debit=0
    for i in range(0,n):
        if(array[i]>=0):
            debit+=1
      
    return debit     

#total no of credit transaction
def num_credit(array):
    n = len(array)
    
    credit=0
    for i in range(0,n):
        if(array[i]<0):
            credit+=1
      
    return credit   


#all debit values
def debit_values(array):
    n = len(array)
    debit_list = []
    for i in range(0,n):
        if(array[i]>=0):
            debit_list.append(array[i])
      
    return debit_list     
        
 #all credit values

def credit_values(array):
    n = len(array)
    credit_list = []
    for i in range(0,n):
        if(array[i] < 0):
            credit_list.append(array[i])
      
    return credit_list


#getting debit and credit values
df['debit_values'] = df.iloc[:,0].apply(lambda x : np.array(debit_values(x))) 
df['credit_values'] = df.iloc[:,0].apply(lambda x : np.array(credit_values(x)))

#creating features distinguishly for debit and credit transactions

df['debit_min_amount'] = df.iloc[:,13].apply(lambda x : np.min(x) if len(x)>0 else 0)
df['debit_max_amount'] = df.iloc[:,13].apply(lambda x : np.max(x) if len(x)>0 else 0)
df['debit_median_amount'] = df.iloc[:,13].apply(lambda x : np.median(x) if len(x)>0 else 0)
df['debit_mean_amount'] = df.iloc[:,13].apply(lambda x : np.mean(x) if len(x)>0 else 0)

df['credit_min_amount'] = df.iloc[:,14].apply(lambda x : np.min(x) if len(x)>0 else 0)
df['credit_max_amount'] = df.iloc[:,14].apply(lambda x : np.max(x) if len(x)>0 else 0)
df['credit_median_amount'] = df.iloc[:,14].apply(lambda x : np.median(x) if len(x)>0 else 0)
df['credit_mean_amount'] = df.iloc[:,14].apply(lambda x : np.mean(x) if len(x)>0 else 0)




#no of debit and credit data
df['no_debit'] = df.iloc[:,0].apply(lambda x : num_debit(x))
df['no_credit'] = df.iloc[:,0].apply(lambda x : num_credit(x))

#dropping unnecessary features
df = df.drop(['debit_values', 'credit_values','transaction_amount', 'days_before_request', 'loan_date'], axis=1)

train_df = df.iloc[0:10000, :]
test_df = df.iloc[10000:,:]
train_df['isDefault'] = train_df.iloc[:,1].apply(lambda x : x[1])

X = train_df.drop('isDefault', axis=1)
y = train_df['isDefault']

#Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Scale Data
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_train=pd.DataFrame(X_train,columns=X.columns)
X_test=sc.transform(X_test)

#Apply RandomForest Algorethm
random_classifier= RandomForestClassifier()
random_classifier.fit(X_train,y_train)
y_pred= random_classifier.predict(X_test)
y_prob= random_classifier.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('area under curve = ' + str(roc_auc))

#get features Importances
xx= pd.Series(random_classifier.feature_importances_,index=X.columns)
xx.sort_values(ascending=False)

accuracy_score(y_test,y_pred)

X_test_data = test_df.drop('isDefault',axis=1)
index = X_test_data.index
X_test_data=sc.transform(X_test_data)
Probabilities = random_classifier.predict_proba(X_test_data)[:,1]
Prob_df = pd.DataFrame(data=Probabilities, columns=['Probability of Default'],index=index)

df_to_CSV = Prob_df.to_csv('/Users/Adamyae/Desktop/Zhiyuan_Ye.csv')