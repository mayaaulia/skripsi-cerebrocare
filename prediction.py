# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import smote_variants as sv
from genetic_selection import GeneticSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn.metrics import accuracy_score
import pickle
import joblib

# loading and reading the dataset
stroke = pd.read_csv("stroke-dataset.csv")
df = stroke.copy()

# Menghapus id
df = df.drop(['id'], axis=1)

# Mengecek missing value
c=df.isnull().sum()
p=round(df.isnull().sum()*100/len(df),2)
# Concatenate the counts and percentages into a new DataFrame 'con_df'
con_df = pd.concat([c, p], axis=1, keys=["counts", "percentage"])
con_df

# Mengatasi missing value dengan rata-rata
#We have only 1 column that have missing values
df["bmi"]=df["bmi"].fillna(df["bmi"].mean())
df["bmi"].isnull().sum()

sns.heatmap(df.isnull(), cmap="summer")
plt.show()

#Mengecek data duplikat
df.duplicated().sum()

# feature encoder
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['ever_married'] = label_encoder.fit_transform(df['ever_married'])
df['Residence_type'] = label_encoder.fit_transform(df['Residence_type'])
df['work_type'] = label_encoder.fit_transform(df['work_type'])
df['smoking_status'] = label_encoder.fit_transform(df['smoking_status'])

# Mengecek data tidak seimbang
ax= sns.countplot(data=df, x="stroke",palette="Set2")
#for bars in ax.containers:
 #   ax.bar_label(bars)
percentages = df['stroke'].value_counts() / len(df) * 100
# Annotate bars with percentages
for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.5, f'{percentages[i]:.2f}%', ha="center")
# Show the plot
plt.show()

X=df.drop(["stroke"], axis=1)
y=df["stroke"]

# Splitting data
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=42)
print("x-train: ", X_train.shape)
print("x-test: ",X_test.shape)
print("y-train: ",y_train.shape)
print("y-test: ",y_test.shape)

# Klasifikasi dengan RFC
clf = RandomForestClassifier( n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred_RF = clf.predict(X_test)

print(classification_report(y_test, y_pred_RF))
ReportRFC = classification_report(y_test, y_pred_RF)
cm = confusion_matrix(y_test, y_pred_RF)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
accuracy = accuracy_score(y_test, y_pred_RF)
print(f'Akurasi: {accuracy}')

# Mengatasi imbalance data dengan SMOTE

# # Inisialisasi SMOTE
# smote = SMOTE(random_state=42)

# # Melakukan oversampling pada data latih
# X_oversampler, y_oversampler = smote.fit_resample(X_train, y_train)

# # Sekarang Anda dapat melatih model menggunakan data yang sudah di-oversample
# print('After OverSampling, the shape of train_X: {}'.format(X_oversampler.shape))
# print('After OverSampling, the shape of train_y: {} \n'.format(y_oversampler.shape))

# print("After OverSampling, counts of label '1': {}".format(sum(y_oversampler == 1)))
# print("After OverSampling, counts of label '0': {}".format(sum(y_oversampler == 0)))
# Menampilkan distribusi data sebelum oversampling
# Menghitung jumlah sampel setiap kelas
class_counts = y_train.value_counts()

# Menghitung presentase setiap kelas
class_percentages = class_counts / len(y_train) * 100

# Menampilkan distribusi data sebelum oversampling
sns.countplot(x=y_train, palette='Set2')

# Menambahkan label presentase ke plot
for i, count in enumerate(class_counts):
    plt.text(i, count, f'{class_percentages[i]:.2f}%', ha='center', va='bottom')

plt.title('Data Distribution Before Oversampling')
plt.show()
# Menghitung jumlah masing-masing nilai atau kelas
class_counts = y_train.value_counts()

# Mencetak jumlah masing-masing nilai atau kelas
print("Jumlah masing-masing nilai:")
print(class_counts)

# Pilih variasi SMOTE yang sesuai
oversampler = sv.MulticlassOversampling(oversampler="SMOTE", oversampler_params={'random_state':42})
X_oversampler, y_oversampler = oversampler.sample(X_train, y_train)

# Sekarang Anda dapat melatih model menggunakan data yang sudah di-SMOTE
print('After OverSampling, the shape of train_X: {}'.format(X_oversampler.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_oversampler.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_oversampler == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_oversampler == 0)))


#Visualisasi data setelah SMOTE
sns.countplot(x=y_oversampler, palette='Set2')
plt.title('Distribusi data setelah SMOTE')
plt.show()

# Seleksi fitur dengan Algortima Genetika
X_train, X_test, y_train, y_test = train_test_split(X_oversampler, y_oversampler, test_size=0.3, random_state=42)

# Estimator Random Forest
estimator = RandomForestClassifier(random_state=42)

ga_feature = GeneticSelectionCV(
    estimator,
    n_population=20,
    crossover_independent_proba=0.75,
    mutation_independent_proba=0.1,
)

# Train and select the features
X_train_ga = ga_feature.fit_transform(X_train, y_train)
X_test_ga = ga_feature.transform(X_test)

# Train model dengan algoritma Random Forest
clf_ga = RandomForestClassifier(n_estimators=100, random_state=42)
clf_ga.fit(X_train_ga, y_train)
y_predict_ga = clf_ga.predict(X_test_ga)

print(classification_report(y_test, y_predict_ga))
cm = confusion_matrix(y_test, y_predict_ga)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

accuracy = accuracy_score(y_test, y_predict_ga)
print(f'Akurasi: {accuracy}')
print("Selesai")