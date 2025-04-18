import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Load data
df = pd.read_csv("Superstore-Data-1-review (1).csv", encoding='windows-1254')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Normalize numerical features
num_features = ['Sales', 'Quantity', 'Discount', 'Profit', 'Rate']
scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# Encode categorical variables
label_encoders = {}
categorical_features = ['Product ID', 'Customer ID']
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save encoders
with open("label_encoders.pkl", 'wb') as f:
    pickle.dump(label_encoders, f)

# Train collaborative filtering model
reader = Reader(rating_scale=(df['Rate'].min(), df['Rate'].max()))
data = Dataset.load_from_df(df[['Customer ID', 'Product ID', 'Rate']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

# Save model and required data
with open("collaborative_model.pkl", 'wb') as f:
    pickle.dump((model, trainset, df), f)

print("✅ Model and encoders saved successfully.")