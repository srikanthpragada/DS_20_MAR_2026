import pickle 
import pandas as pd

f = open("lr_model.pickle","rb")
model = pickle.load(f)  # Unpickle

gre = int(input("Enter GRE   : "))
tof = int(input("Enter Toefl : "))
cgpa = float(input("Enter CGPA : "))
df = pd.DataFrame([[gre,tof,cgpa]], columns = ['Gre','Toefl','Cgpa'])
result = model.predict(df)
print(f"{result[0]:.2f}")
