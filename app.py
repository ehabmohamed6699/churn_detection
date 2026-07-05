import streamlit as st
import joblib
import sklearn

model = joblib.load("AdaBoostClassifier.pkl")
scaler = joblib.load("Standarizer.pkl")

st.title("Churn Detection")
st.text("This project detects and expects churn from users data")

age = st.number_input("Age: ", min_value=0, max_value=100)
gender = st.radio("Gender: ", ["Male", "Female"])
tenure = st.number_input("Tenure: ", min_value=0, max_value=100)
freq = st.number_input("Usage Frequency: ", min_value=0, max_value=100)
support_calls = st.number_input("Support Calls No.: ", min_value=0, max_value=100)
delay = st.number_input("Payment Delay: ", min_value=0, max_value=100)
sub_type = st.radio("Subscription Type: ", ["Basic", "Standard", "Premium"])
contract_len = st.radio("Contract Length: ", ["Monthly", "Quarterly", "Annual"])
total_spend = st.number_input("Total Spend: ", min_value=0, max_value=10000)
last_interaction = st.number_input("Last Interaction: ", min_value=0, max_value=100)

if st.button("Check"):
    gender = 0 if gender == "Male" else 1
    sub_type = 0 if sub_type == "Basic" else 1 if "Standard" else 2
    contract_len = 0 if contract_len == "Monthly" else 1 if "Quarterly" else 2
    X = [[age, gender, tenure, freq, support_calls, delay, sub_type, contract_len, total_spend, last_interaction]]
    X = scaler.transform(X)
    y_pred = model.predict(X)
    if y_pred[0] == 1.0:
        st.warning("Expecting Churn!")
    else:
        st.success("Not Expecting Churn!")