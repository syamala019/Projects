import streamlit as st
import pandas as pd

st.header("Insurance_Claim")

st.write("Predictive Model Built on Below Sample Data")

data = pd.read_csv("modified_insurance data.csv")
st.dataframe(data.head())

col1,col2 = st.columns(2)
with col1:
    kidsdriv = st.number_input(f"Enter the number of kids of driver Min {data.KIDSDRIV.min()} to Max {data.KIDSDRIV.max()}")
    age = st.number_input(f"Enter Age Min {data.AGE.min()} to Max {data.AGE.max()}")
    yoj = st.number_input(f"Enter years on job Min {data.YOJ.min()} to Max {data.YOJ.max()}")
    marital = st.selectbox("Enter the Marital Status:",data.MSTATUS.unique())
    education = st.selectbox("Enter education level:",data.EDUCATION.unique())
    bluebook = st.number_input(f"Enter Bluebook value of car Min {data.BLUEBOOK.min()} to Max {data.BLUEBOOK.max()}")
    cartype = st.selectbox("Enter Type of Car:",data.CAR_TYPE.unique())
    oldclaim = st.number_input(f"Enter the old claim amount Min {data.OLDCLAIM.min()} to Max {data.OLDCLAIM.max()}")
    revoked = st.selectbox("Has the license been revoked:",data.REVOKED.unique())
    clmamt = st.number_input(f"Enter the claim amount Min {data.CLM_AMT.min()} to Max {data.CLM_AMT.max()}")
    city = st.selectbox("Enter urbanicity:",data.URBANICITY.unique())
with col2:
    birth = st.number_input(f"Enter year of birth Min {data.BIRTH.min()} to Max {data.BIRTH.max()}")
    homekids= st.number_input(f"Enter number of home kids Min {data.HOMEKIDS.unique()} to Max {data.HOMEKIDS.max()}")
    income = st.number_input(f"Enter Income Min {data.INCOME.min()} to Max {data.INCOME.max()}")
    homeval = st.number_input(f"Enter Home Value Min {data.HOME_VAL.min()} to Max {data.HOME_VAL.max()}")
    gender = st.selectbox("Enter the Gender:",data.GENDER.unique())
    occupation = st.selectbox("Enter the Occupation:",data.OCCUPATION.unique())
    caruse = st.selectbox("Enter purpose of Car Used:",data.CAR_USE.unique())
    tif= st.number_input(f"Enter duration of Insurance Min {data.TIF.min()} to Max {data.TIF.max()}")
    freq = st.number_input(f"Enter claim frequency Min {data.CLM_FREQ.min()} to Max {data.CLM_FREQ.max()}")
    mvr = st.number_input(f"Enter Vehicle Record Points Min {data.MVR_PTS.min()} to Max {data.MVR_PTS.max()}")
    carage = st.number_input(f"Enter Age of the Car Min {data.CAR_AGE.min()} to Max {data.CAR_AGE.max()}")


xdata = [kidsdriv,birth,age,homekids,yoj,income,homeval,marital,gender,education,occupation,
         caruse,bluebook,tif,cartype,oldclaim,freq,revoked,mvr,clmamt,carage,city]

import joblib
with open('model.pkl','rb') as f:
    model = joblib.load(f)
with open('encoder.pkl','rb') as p:
    encoder = joblib.load(p)
with open('ordinal.pkl','rb') as k:
    ordinal = joblib.load(k)
with open('scaler.pkl','rb') as s:
    scaler = joblib.load(s)

x = pd.DataFrame([xdata], columns=data.columns[0:22])

st.write("Given Input:")
st.dataframe(x)

# Applying one-hot encoding to the nominal columns
categorical_columns = ['OCCUPATION','CAR_TYPE']
input_encoded = encoder.transform(x[categorical_columns])
input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Drop original categorical columns and concatenate the encoded values
x = x.drop(columns=categorical_columns)
x = pd.concat([x, input_encoded_df], axis=1)

# Applying the ordinal encoding to the ordinal column 
x['EDUCATION'] = ordinal.fit_transform(x[['EDUCATION']])

if isinstance(x, pd.DataFrame):  # Check if xtrain is a Pandas DataFrame
    x.columns = x.columns.astype(str)
    x.columns = x.columns.str.replace(r'[<>[\]]', '', regex=True)

replacement_mappings = {
'MSTATUS': {'yes': 1, 'z_no': 0},
'GENDER': {'m': 1, 'z_f': 0},
'CAR_USE': {'commercial': 1, 'private': 0},
'REVOKED': {'yes': 1, 'no': 0},
'URBANICITY': {'highly urban/ urban': 1, 'z_highly rural/ rural': 0}}

for column, mapping in replacement_mappings.items():
    if column in x.columns:
        x[column] = x[column].replace(mapping)

if st.button("Predict"):
    prediction = model.predict(x)

    if prediction == 1:
        st.write("Claimed")
    else:
        st.write("Not Claimed")