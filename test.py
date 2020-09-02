from sklearn.ensemble import ExtraTreesClassifier
from ForwardFeatureSelection import ForwardFeatureSelection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


df = pd.read_csv("/home/dckap/Documents/FeatureSelection/Iris.csv")
et = ExtraTreesClassifier()

# print(isinstance(None, object))
df = df[['SepalLengthCm', 'SepalWidthCm',
         'PetalLengthCm', 'PetalWidthCm', 'Species']]

ffs = ForwardFeatureSelection(
    et, df, 'Species',  scale_obj=MinMaxScaler(), max_no_features=3, min_no_features=2)
# print(dir(ffs))

print(ffs.run())
