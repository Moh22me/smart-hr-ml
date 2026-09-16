import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="HR Analytics & ML Predictor", layout="wide", page_icon="📊")

# ==========================================
# 2. Efficiently Load Data and Model
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv('data/hr_data.csv')

@st.cache_resource
def load_model_components():
    model = joblib.load('models/rf_model.pkl')
    encoders = joblib.load('models/encoders.pkl')
    features = joblib.load('models/features.pkl')
    return model, encoders, features

df = load_data()
model, encoders, features = load_model_components()

# ==========================================
# 3. Dashboard Header
# ==========================================
st.title("📊 Smart HR Analytics & Attrition Predictor")
st.markdown("Machine Learning Dashboard for Exploratory Data Analysis (EDA) and Employee Attrition Classification.")

# Create Tabs
tab1, tab2 = st.tabs(["📈 Data Analysis (EDA)", "🤖 Predict Attrition (ML Model)"])

# ==========================================
# TAB 1: Exploratory Data Analysis (EDA)
# ==========================================
with tab1:
    st.header("📈 Exploratory Data Analysis (EDA)")
    st.write("Use the interactive filters below to analyze employee trends dynamically:")

    # Interactive Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        department_filter = st.multiselect("Filter by Department:", options=df['Department'].unique(), default=df['Department'].unique())
    with col_f2:
        gender_filter = st.multiselect("Filter by Gender:", options=df['Gender'].unique(), default=df['Gender'].unique())
    with col_f3:
        age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
        selected_age = st.slider("Filter by Age Range:", age_min, age_max, (age_min, age_max))

    # Apply Filters to DataFrame
    filtered_df = df[
        (df['Department'].isin(department_filter)) &
        (df['Gender'].isin(gender_filter)) &
        (df['Age'].between(selected_age[0], selected_age[1]))
    ]

    st.markdown(f"**Showing {len(filtered_df)} out of {len(df)} employees based on selected filters.**")
    st.dataframe(filtered_df.head(), use_container_width=True)

    st.markdown("---")

    # Dynamic Visualizations
   # Dynamic Visualizations (Guaranteed to update!)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition Count (Filtered)")
        fig, ax = plt.subplots(figsize=(6, 4))
        attrition_counts = filtered_df['Attrition'].value_counts()
        sns.barplot(x=attrition_counts.index, y=attrition_counts.values, palette=['#4CAF50', '#FF5252'], ax=ax)
        ax.set_title("Count of Staying (No) vs Leaving (Yes)")
        ax.set_ylabel("Number of Employees")
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Income vs Attrition")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=filtered_df, x='Attrition', y='MonthlyIncome', palette='Set2', ax=ax)
        ax.set_title("Income Salary Distribution")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Job Satisfaction vs Attrition")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x='JobSatisfaction', hue='Attrition', palette='coolwarm', ax=ax)
        ax.set_title("Satisfaction Levels (1=Low, 4=High)")
        st.pyplot(fig)

    with col4:
        st.subheader("OverTime Impact on Attrition")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x='OverTime', hue='Attrition', palette='magma', ax=ax)
        ax.set_title("OverTime Work Analysis")
        st.pyplot(fig)

# ==========================================
# TAB 2: ML Prediction Engine
# ==========================================
with tab2:
    st.header("🤖 Predict Employee Attrition")
    st.write("Adjust the features below to test the Random Forest Classifier.")

    # User Inputs
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        age = st.slider("Age", 18, 60, 30)
        monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
        job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4], index=2)
        
    with col_p2:
        over_time = st.selectbox("OverTime", ["Yes", "No"])
        total_working_years = st.slider("Total Working Years", 0, 40, 5)
        years_at_company = st.slider("Years At Company", 0, 40, 3)

    if st.button("Run ML Model 🚀", type="primary"):
        # 1. Start with safe median/mode values for all background features
        input_data = {}
        for col in features:
            if pd.api.types.is_numeric_dtype(df[col]):
                input_data[col] = df[col].median()
            else:
                input_data[col] = df[col].mode()[0]
        
        # 2. Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # 3. Inject User Specific Values
        input_df.at[0, 'Age'] = age
        input_df.at[0, 'MonthlyIncome'] = monthly_income
        input_df.at[0, 'JobSatisfaction'] = job_satisfaction
        input_df.at[0, 'OverTime'] = over_time
        input_df.at[0, 'TotalWorkingYears'] = total_working_years
        input_df.at[0, 'YearsAtCompany'] = years_at_company

        # 4. Apply Pre-trained Encoders safely
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        # 5. Ensure Order matches Training Data
        input_df = input_df[features]

        # 6. Execute Prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        stay_prob = probabilities[0] * 100
        leave_prob = probabilities[1] * 100

        # 7. Display Results cleanly
        st.markdown("---")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(label="Probability of Staying (Low Risk)", value=f"{stay_prob:.1f}%")
        with col_res2:
            st.metric(label="Probability of Leaving (High Risk)", value=f"{leave_prob:.1f}%")

        if prediction == 1:
            st.error(f"⚠️ **Classification: WILL LEAVE (High Risk)**\nThe employee shows significant risk factors for attrition.")
        else:
            st.success(f"✅ **Classification: WILL STAY (Low Risk)**\nThe employee profile aligns with stable retention trends.")