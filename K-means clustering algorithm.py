#install Kaggle
!pip install -q Kaggle

from google.colab import files
files.upload()
#create a kaggle folder
!mkdir ~/.kaggle
#copy the kaggle json file to folder created
! cp kaggle.json ~/.kaggle/
#permission for json to act
! chmod 600 ~/.kaggle/kaggle.json
#to list all dataset in kaggle
! kaggle datasets list

!kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("/content/customer-segmentation-tutorial-in-python.zip")
data.head()
sns.scatterplot(x=data['Annual Income (k$)'],y=data['Spending Score (1-100)'])
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5)
kmeans.fit(data[['Annual Income (k$)','Spending Score (1-100)']])
labels=kmeans.labels_

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Mall Customer Segmentation')

sns.scatterplot(x=data['Annual Income (k$)'],y=data['Spending Score (1-100)'],hue=labels,palette='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', s=100)
data['Cluster']=kmeans.predict(data[['Annual Income (k$)','Spending Score (1-100)']])
data.head()
