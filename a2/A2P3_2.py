import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

#generate data from tsv files
data = pd.read_csv("./data/data.tsv", sep = "\t")

#--------------overfit--------------------
#25 with label 0, 25 with label 1
overfit_0 = data[data['label'] == 0].sample(n=25, random_state=41)
overfit_1 = data[data['label'] == 1].sample(n=25, random_state=41)

#cancatenate & shuffle
overfit = sklearn.utils.shuffle(pd.concat([overfit_0, overfit_1]))

#--------------train, test, validation--------------------
#use <stratify> to ensure equal label in all datasets
x = data['text']
y = data['label']

#test = 20% of total data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, shuffle=True, random_state=41)

#train = 80%*80% = 64% of total data
#validation = 80%*20% = 16% of total data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, shuffle=True, random_state=41)

#combine x and y
train = pd.concat([x_train, y_train], axis=1)
validation = pd.concat([x_val, y_val], axis=1)
test = pd.concat([x_test, y_test], axis=1)

#write into tsv files
train.to_csv("./data/train.tsv", sep="\t")
validation.to_csv("./data/validation.tsv", sep="\t")
test.to_csv("./data/test.tsv", sep="\t")
overfit.to_csv("./data/overfit.tsv", sep="\t")

#check equal label distribution in each dataset
print("Checking label distribution in train dataset")
print(train["label"].value_counts())

print("Checking label distribution in validation dataset")
print(validation["label"].value_counts())

print("Checking label distribution in test dataset")
print(test["label"].value_counts())

print("Checking label distribution in overfit dataset")
print(overfit["label"].value_counts())

#make sure there is no same sample used more than one of the dataset
#concatenate all 3 datasets, drop duplicates and check size
concatenated = pd.concat([train, validation, test])
concatenated = concatenated.drop_duplicates()
print(f"If we concatenate train, validation, and test, drop duplicates(if applies...)")
print(f"the length is: {len(concatenated)}")