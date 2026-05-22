## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('data.csv')
df
```
<img width="885" height="522" alt="Screenshot 2026-05-22 113348" src="https://github.com/user-attachments/assets/abb4db12-bc51-4c56-93c7-dc44e8749482" />

```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
climate = ['Cold','Warm','Hot','Very Hot']
ele = OrdinalEncoder(categories=[climate])
ele.fit_transform(df[["Ord_1"]])
```
<img width="1400" height="357" alt="Screenshot 2026-05-22 113435" src="https://github.com/user-attachments/assets/e7c3b840-c94e-4634-a8a3-43d464eaa92f" />

```
df['bo2'] = ele.fit_transform(df[["Ord_1"]])
df
```
<img width="1401" height="452" alt="Screenshot 2026-05-22 113444" src="https://github.com/user-attachments/assets/82d7b367-e690-4dab-a849-dd78856087f6" />

```
le = LabelEncoder()
df2 = df.copy()
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```
<img width="1407" height="496" alt="Screenshot 2026-05-22 113455" src="https://github.com/user-attachments/assets/4fe0d0a4-2998-4f3f-ac4e-142af5efe3e4" />

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
df3 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["City"]]))
df2 = pd.concat([enc,df3],axis = 1)
df2
```
<img width="1398" height="531" alt="Screenshot 2026-05-22 113506" src="https://github.com/user-attachments/assets/20724938-b5d5-468c-845b-d778b4e177bb" />

```
pd.get_dummies(df,columns=['City'])
```
<img width="1401" height="432" alt="Screenshot 2026-05-22 113513" src="https://github.com/user-attachments/assets/1504dd50-ec3a-41bd-a4af-4b3dee64db9c" />

```
from category_encoders import BinaryEncoder
df = pd.read_csv('data.csv')
df
```
<img width="1403" height="487" alt="Screenshot 2026-05-22 113521" src="https://github.com/user-attachments/assets/fd51fa9c-6ff8-4cee-b849-7708e4765ffe" />

```
be = BinaryEncoder()
nd = be.fit_transform(df['Ord_2'])
df
```
<img width="1399" height="486" alt="Screenshot 2026-05-22 113528" src="https://github.com/user-attachments/assets/c5cf0388-5100-428b-8f46-c488640b63e4" />

```
from category_encoders import TargetEncoder
te = TargetEncoder()
CC = df.copy()
new = te.fit_transform(CC["City"],y=CC["Target"])
CC = pd.concat([CC,new],axis = 1)
CC
```
<img width="1401" height="551" alt="Screenshot 2026-05-22 113537" src="https://github.com/user-attachments/assets/4259fdeb-3586-45cf-a233-960d1ecbe79b" />

```
if 'City' in CC.columns:
    CC = CC.drop('City', axis=1)
new = te.fit_transform(X = df["City"],y=df["Target"])
CC = pd.concat([CC.reset_index(drop=True),new.reset_index(drop=True)],axis = 1)
CC
```
<img width="1405" height="530" alt="Screenshot 2026-05-22 113546" src="https://github.com/user-attachments/assets/7f1ea809-2e9d-4d7e-8ede-e3663f193e9a" />

```
df = pd.read_csv('Data_to_Transform.csv')
df
```
<img width="1396" height="540" alt="Screenshot 2026-05-22 113554" src="https://github.com/user-attachments/assets/226b1592-e236-4dfc-82fa-eece1718b8c3" />

```
df.skew()
```
<img width="1394" height="182" alt="Screenshot 2026-05-22 113605" src="https://github.com/user-attachments/assets/a5c0384c-3384-411a-9560-19e7a50aa763" />

```
np.log(df["Highly Positive Skew"])
```
<img width="1397" height="333" alt="Screenshot 2026-05-22 113612" src="https://github.com/user-attachments/assets/9118b162-056a-40aa-85b3-f82495255781" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="1391" height="333" alt="Screenshot 2026-05-22 113621" src="https://github.com/user-attachments/assets/8cad5596-1759-405a-a79b-f923cf424595" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="1397" height="329" alt="Screenshot 2026-05-22 113632" src="https://github.com/user-attachments/assets/1da45193-d2d7-4cc9-93ab-6c41ed4bbc93" />

```
np.square(df["Highly Positive Skew"])
```
<img width="1397" height="333" alt="Screenshot 2026-05-22 113644" src="https://github.com/user-attachments/assets/0837d6ca-649c-4990-bcf0-e8e0e77a38dc" />

```
df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1399" height="541" alt="Screenshot 2026-05-22 113657" src="https://github.com/user-attachments/assets/ef9ccac6-d2dc-4179-a977-33467f6fc25d" />

```
df["Moderate Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Moderate Negative Skew"])
df
```
<img width="1398" height="558" alt="Screenshot 2026-05-22 113706" src="https://github.com/user-attachments/assets/709bbf7c-9a15-4aea-b76b-5d7ff4206a5f" />

```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1405" height="599" alt="Screenshot 2026-05-22 113719" src="https://github.com/user-attachments/assets/d9391248-af52-427d-a027-0b74c3affa03" />

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line = '45')
plt.show()
```
<img width="1395" height="721" alt="Screenshot 2026-05-22 113731" src="https://github.com/user-attachments/assets/538db9d5-9523-4c68-883e-9e116ad2961a" />

```
sm.qqplot(df["Moderate Negative Skew_1"],line = '45')
plt.show()
```
<img width="1393" height="637" alt="Screenshot 2026-05-22 113741" src="https://github.com/user-attachments/assets/0c0ffde8-e2ec-4b6c-a5f0-029cb486482d" />

```
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line = '45')
plt.show()
```
<img width="1394" height="659" alt="Screenshot 2026-05-22 113753" src="https://github.com/user-attachments/assets/182c82a1-9400-43ef-b07c-877c226c3eea" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line = '45')
plt.show()
```
<img width="1399" height="633" alt="Screenshot 2026-05-22 113804" src="https://github.com/user-attachments/assets/b269310d-1a17-459c-902f-b5d6cb6432aa" />

```
sm.qqplot(df["Highly Negative Skew_1"],line = '45')
plt.show()
```
<img width="1397" height="638" alt="Screenshot 2026-05-22 113816" src="https://github.com/user-attachments/assets/1466ee32-2cfc-4197-adff-52fac324148e" />

```
sm.qqplot(np.abs(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```
<img width="1395" height="636" alt="Screenshot 2026-05-22 113838" src="https://github.com/user-attachments/assets/bb0554f5-6d43-444c-bab7-026d23d88bc8" />

```
sm.qqplot(np.log(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```
<img width="1402" height="705" alt="Screenshot 2026-05-22 113847" src="https://github.com/user-attachments/assets/bd487a82-cee3-4d87-9184-cacd62a0f27c" />

```
sm.qqplot(np.sqrt(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
<img width="1395" height="692" alt="Screenshot 2026-05-22 113858" src="https://github.com/user-attachments/assets/d016a538-084e-4b5b-affa-dca4788c90d9" />

```
pd.concat([CC,new],axis = 1)
```
<img width="1400" height="442" alt="Screenshot 2026-05-22 113912" src="https://github.com/user-attachments/assets/726e363e-0b2e-4a6a-96aa-2ead2fdb051b" />

# RESULT:

  Thus, the given dataset was successfully cleaned, encoded using various Feature Encoding techniques, transformed using different Transformation methods, and saved to a file successfully.   
