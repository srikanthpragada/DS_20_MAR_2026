import pandas as pd

model = pd.read_pickle("lr_model.pickle")   # unpickle model 

gre = int(input("Enter GRE   : "))
tof = int(input("Enter Toefl : "))
cgpa = float(input("Enter CGPA : "))

df = pd.DataFrame([[gre,tof,cgpa]], columns = ['Gre','Toefl','Cgpa'])
result = model.predict(df)
print(f"{result[0]:.2f}")
