"""
Smoking Cessation Prediction Dashboard

A web application presenting research findings and providing personalized
quit method recommendations based on the PATH Study longitudinal data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from modeling import load_model
from feature_engineering import get_feature_list

# Page configuration
st.set_page_config(
    page_title="Smoking Cessation Research Dashboard",
    page_icon="🚭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and data
@st.cache_resource
def load_resources():
    """Load model and feature importance data."""
    model, _ = load_model(str(PROJECT_ROOT / 'models' / 'xgboost_best.pkl'))
    
    # Get feature importance
    importance = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v} 
        for k, v in importance.items()
    ]).sort_values('importance', ascending=False)
    
    return model, importance_df

model, importance_df = load_resources()

# Sidebar navigation
st.sidebar.title("🚭 Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["📊 Research Findings", "🎯 Cessation Quiz", "ℹ️ About"]
)

# ============================================================================
# PAGE 1: RESEARCH FINDINGS
# ============================================================================
if page == "📊 Research Findings":
    st.title("🔬 Smoking Cessation Research Findings")
    st.markdown("### Based on PATH Study Waves 1-7 (2013-2020)")
    
    st.markdown("---")
    
    # Key findings summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Study Participants", "24,576", help="Unique individuals tracked")
    
    with col2:
        st.metric("Quit Attempts Analyzed", "59,984", help="Person-period transitions")
    
    with col3:
        st.metric("Model Accuracy (AUC)", "0.87", help="Prediction performance")
    
    st.markdown("---")
    
    # Top predictive factors
    st.subheader("🎯 Top Factors Predicting Quit Success")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Feature importance chart
        top_10 = importance_df.head(10)
        fig = go.Figure(go.Bar(
            x=top_10['importance'],
            y=top_10['feature'],
            orientation='h',
            marker=dict(color='steelblue')
        ))
        fig.update_layout(
            title="Most Important Predictive Features",
            xaxis_title="Importance Score",
            yaxis_title="",
            height=400,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("#### 🔑 Key Insights")
        st.markdown("""
        **Light Smoking (<10 cpd)**
        - Strongest predictor of success
        - 2x higher quit rates
        
        **High Dependence**
        - Major barrier to quitting
        - Early morning smoking = higher dependence
        
        **Time to First Cigarette**
        - Measures addiction severity
        - <5 minutes = very high dependence
        """)
    
    st.markdown("---")
    
    # Evidence-based interventions
    st.subheader("💊 Evidence-Based Quit Methods")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Varenicline (Chantix)")
        st.success("✅ Highly Effective")
        st.markdown("""
        - Prescription medication
        - Reduces cravings
        - Blocks nicotine receptors
        - **Most effective pharmacotherapy**
        """)
    
    with col2:
        st.markdown("#### Nicotine Replacement (NRT)")
        st.info("✅ Effective")
        st.markdown("""
        - Patches, gum, lozenges
        - Over-the-counter
        - Reduces withdrawal
        - Best combined with counseling
        """)
    
    with col3:
        st.markdown("#### Behavioral Counseling")
        st.info("✅ Effective")
        st.markdown("""
        - One-on-one or group
        - Phone quitlines available
        - Addresses triggers
        - Enhances medication effects
        """)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("📋 Clinical Recommendations")
    
    st.markdown("""
    Based on analysis of 59,984 quit attempts:
    
    1. **Screen for Dependence Level**
       - Use TTFC (time to first cigarette) and CPD (cigarettes per day)
       - Tailor intervention intensity to dependence level
    
    2. **Prioritize Evidence-Based Methods**
       - Offer varenicline as first-line for high-dependence smokers
       - Combine NRT with behavioral support
       - Avoid cold turkey for highly dependent smokers
    
    3. **Address Environmental Factors**
       - Encourage smokefree home policies
       - Consider household composition
       - Build supportive quit environment
    
    4. **Leverage Motivation**
       - Assess readiness to quit
       - Time interventions with high motivation
       - Set concrete quit dates
    """)

# ============================================================================
# PAGE 2: CESSATION QUIZ
# ============================================================================
elif page == "🎯 Cessation Quiz":
    st.title("🎯 Personalized Cessation Method Recommender")
    st.markdown("### Find the quit method with the highest success probability for you")
    
    st.markdown("---")
    
    st.markdown("""
    Answer the questions below to receive a personalized recommendation based on 
    longitudinal data from over 24,000 smokers in the PATH Study.
    """)
    
    with st.form("cessation_quiz"):
        st.subheader("📝 Your Smoking Habits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cigarettes per day
            cpd = st.number_input(
                "Cigarettes per day (on average)",
                min_value=1,
                max_value=60,
                value=10,
                help="How many cigarettes do you typically smoke per day?"
            )
            
            # Time to first cigarette
            ttfc_options = {
                "Within 5 minutes": 2.5,
                "6-30 minutes": 20,
                "31-60 minutes": 45,
                "After 60 minutes": 120
            }
            ttfc_label = st.selectbox(
                "How soon after waking do you smoke?",
                options=list(ttfc_options.keys()),
                help="Time to first cigarette is a strong indicator of dependence"
            )
            ttfc = ttfc_options[ttfc_label]
            
            # Previous quit attempts
            previous_quits = st.number_input(
                "Number of previous quit attempts",
                min_value=0,
                max_value=20,
                value=1,
                help="How many times have you tried to quit in the past?"
            )
        
        with col2:
            # Demographics
            age = st.number_input(
                "Your age",
                min_value=18,
                max_value=90,
                value=35
            )
            
            # Education
            education_options = {
                "Less than high school": 0,
                "High school graduate": 0,
                "Some college": 0,
                "College degree or higher": 1
            }
            education_label = st.selectbox(
                "Highest education level",
                options=list(education_options.keys())
            )
            college_degree = education_options[education_label]
            
            # Income
            income_options = {
                "Less than $25,000": 0,
                "$25,000 - $49,999": 0,
                "$50,000 - $99,999": 1,
                "$100,000 or more": 1
            }
            income_label = st.selectbox(
                "Annual household income",
                options=list(income_options.keys())
            )
            high_income = income_options[income_label]
        
        st.subheader("🏠 Your Environment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            smokefree_home = st.radio(
                "Is your home smokefree?",
                options=["Yes", "No"],
                help="Do you have rules against smoking inside your home?"
            )
            smokefree_home = 1 if smokefree_home == "Yes" else 0
            
            household_smokers = st.number_input(
                "Number of other smokers in household",
                min_value=0,
                max_value=10,
                value=0
            )
        
        with col2:
            motivation = st.slider(
                "How motivated are you to quit? (1-10)",
                min_value=1,
                max_value=10,
                value=7,
                help="1 = Not motivated at all, 10 = Extremely motivated"
            )
            
            plans_to_quit = st.radio(
                "Do you plan to quit in the next 30 days?",
                options=["Yes", "No", "Not sure"]
            )
            plans_to_quit = 1 if plans_to_quit == "Yes" else 0
        
        submitted = st.form_submit_button("🔍 Get Personalized Recommendations")
    
    if submitted:
        st.markdown("---")
        st.subheader("📊 Your Personalized Results")
        
        # Calculate dependence level
        high_dependence = 1 if (cpd >= 10 or ttfc <= 30) else 0
        very_high_dependence = 1 if (cpd >= 20 or ttfc <= 5) else 0
        cpd_light = 1 if cpd < 10 else 0
        cpd_heavy = 1 if cpd >= 20 else 0
        age_young = 1 if age < 25 else 0
        motivation_high = 1 if motivation >= 7 else 0
        
        # Create feature vector for different quit methods
        base_features = {
            'high_dependence': high_dependence,
            'very_high_dependence': very_high_dependence,
            'cpd': cpd,
            'cpd_heavy': cpd_heavy,
            'cpd_light': cpd_light,
            'dependence_score': 2 if very_high_dependence else (1 if high_dependence else 0),
            'ttfc_minutes': ttfc,
            'age': age,
            'age_young': age_young,
            'female': 0.5,  # Unknown, use average
            'college_degree': college_degree,
            'high_income': high_income,
            'used_nrt': 0,
            'used_patch': 0,
            'used_gum': 0,
            'used_lozenge': 0,
            'used_varenicline': 0,
            'used_bupropion': 0,
            'used_any_medication': 0,
            'used_counseling': 0,
            'used_quitline': 0,
            'used_any_behavioral': 0,
            'used_any_method': 0,
            'cold_turkey': 1,
            'med_plus_counseling': 0,
            'nrt_plus_med': 0,
            'nrt_plus_counseling': 0,
            'nrt_plus_quitline': 0,
            'med_plus_quitline': 0,
            'num_previous_quits': previous_quits,
            'previous_quit_success': 1 if previous_quits > 0 else 0,
            'longest_quit_duration': np.nan,
            'last_quit_duration_days': np.nan,
            'longest_quit_duration_days': np.nan,
            'recent_vs_longest_ratio': np.nan,
            'motivation_high': motivation_high,
            'plans_to_quit': plans_to_quit,
            'quit_timeframe_code': np.nan,
            'early_quit_intent': plans_to_quit,
            'household_smokers': household_smokers,
            'smokefree_home': smokefree_home,
            'nrt_days_raw': np.nan,
            'rx_days_raw': np.nan,
            'nrt_days_log': np.nan,
            'rx_days_log': np.nan,
            'highdep_x_varenicline': 0,
            'highdep_x_nrt': 0,
            'young_x_counseling': 0,
            'race_white': 0.5,
            'race_black': 0.25,
            'race_hispanic': 0.15,
            'race_other': 0.1
        }
        
        # Test different quit methods
        methods = []
        
        # 1. Varenicline + Counseling
        features_varenicline = base_features.copy()
        features_varenicline.update({
            'used_varenicline': 1,
            'used_counseling': 1,
            'used_any_medication': 1,
            'used_any_behavioral': 1,
            'used_any_method': 1,
            'med_plus_counseling': 1,
            'cold_turkey': 0,
            'highdep_x_varenicline': high_dependence
        })
        
        # 2. NRT (Patch) + Counseling
        features_nrt = base_features.copy()
        features_nrt.update({
            'used_nrt': 1,
            'used_patch': 1,
            'used_counseling': 1,
            'used_any_behavioral': 1,
            'used_any_method': 1,
            'nrt_plus_counseling': 1,
            'cold_turkey': 0,
            'highdep_x_nrt': high_dependence
        })
        
        # 3. Bupropion + Counseling
        features_bupropion = base_features.copy()
        features_bupropion.update({
            'used_bupropion': 1,
            'used_counseling': 1,
            'used_any_medication': 1,
            'used_any_behavioral': 1,
            'used_any_method': 1,
            'med_plus_counseling': 1,
            'cold_turkey': 0
        })
        
        # 4. Counseling only
        features_counseling = base_features.copy()
        features_counseling.update({
            'used_counseling': 1,
            'used_any_behavioral': 1,
            'used_any_method': 1,
            'cold_turkey': 0,
            'young_x_counseling': age_young
        })
        
        # 5. Cold turkey
        features_cold_turkey = base_features.copy()
        
        # Get predictions for each method
        feature_cols = get_feature_list()
        
        for method_name, features in [
            ("Varenicline + Counseling", features_varenicline),
            ("NRT (Patch) + Counseling", features_nrt),
            ("Bupropion (Zyban) + Counseling", features_bupropion),
            ("Counseling Only", features_counseling),
            ("Cold Turkey (No Support)", features_cold_turkey)
        ]:
            # Create DataFrame with features in correct order
            X = pd.DataFrame([features])[feature_cols]
            
            # Predict
            prob = model.predict_proba(X)[0, 1]
            methods.append({
                'method': method_name,
                'success_probability': prob * 100
            })
        
        # Sort by probability
        methods_df = pd.DataFrame(methods).sort_values('success_probability', ascending=False)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart
            fig = go.Figure(go.Bar(
                x=methods_df['success_probability'],
                y=methods_df['method'],
                orientation='h',
                marker=dict(
                    color=methods_df['success_probability'],
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=100
                ),
                text=[f"{p:.1f}%" for p in methods_df['success_probability']],
                textposition='auto'
            ))
            fig.update_layout(
                title="Predicted Quit Success Probability by Method",
                xaxis_title="Success Probability (%)",
                yaxis_title="",
                height=400,
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### 🏆 Best Method for You")
            best_method = methods_df.iloc[0]
            st.success(f"**{best_method['method']}**")
            st.metric(
                "Success Probability",
                f"{best_method['success_probability']:.1f}%"
            )
            
            # Personalized insights
            st.markdown("#### 💡 Key Factors")
            if cpd_light:
                st.markdown("✅ Light smoker advantage")
            if high_dependence or very_high_dependence:
                st.markdown("⚠️ High dependence detected")
            if motivation >= 7:
                st.markdown("✅ Strong motivation")
            if smokefree_home:
                st.markdown("✅ Supportive environment")
            if household_smokers > 0:
                st.markdown("⚠️ Household smokers present")
        
        st.markdown("---")
        
        # Detailed recommendations
        st.subheader("📋 Personalized Action Plan")
        
        if very_high_dependence:
            st.warning("""
            **High Dependence Detected**
            
            Your smoking pattern indicates high nicotine dependence. We strongly recommend:
            - Consult with a healthcare provider about **varenicline (Chantix)** or combination therapy
            - Consider using **counseling** alongside medication
            - Avoid attempting to quit cold turkey
            - Set a quit date 1-2 weeks out to prepare
            """)
        elif high_dependence:
            st.info("""
            **Moderate Dependence**
            
            Your dependence level suggests you would benefit from:
            - **NRT (nicotine patch or gum)** to manage withdrawal
            - **Behavioral counseling** or quitline support (1-800-QUIT-NOW)
            - Gradual reduction may be easier than abrupt quitting
            """)
        else:
            st.success("""
            **Lower Dependence**
            
            You have favorable conditions for quitting:
            - **Counseling** alone may be sufficient
            - Consider NRT for extra support during difficult moments
            - Your lighter smoking habit gives you an advantage
            """)
        
        # Additional resources
        st.markdown("---")
        st.subheader("🔗 Additional Resources")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **National Quitline**
            - Call: 1-800-QUIT-NOW
            - Free counseling
            - Available in all states
            """)
        
        with col2:
            st.markdown("""
            **Smokefree.gov**
            - Free quit plan
            - Text support program
            - Mobile apps
            """)
        
        with col3:
            st.markdown("""
            **Healthcare Provider**
            - Prescription medications
            - Personalized advice
            - Insurance coverage info
            """)

# ============================================================================
# PAGE 3: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Dashboard")
    
    st.markdown("""
    ### Study Background
    
    This dashboard presents findings from an analysis of the **Population Assessment of 
    Tobacco and Health (PATH) Study**, a nationally representative longitudinal study 
    of tobacco use in the United States.
    
    #### Data Source
    - **Study**: PATH Study Waves 1-7 (2013-2020)
    - **Sample**: 24,576 adult smokers
    - **Quit Attempts**: 59,984 person-period observations
    - **Follow-up**: Annual waves over 7 years
    
    #### Methodology
    - **Model**: XGBoost gradient boosting classifier
    - **Features**: 52 behavioral, demographic, and intervention variables
    - **Performance**: 0.87 AUC (area under ROC curve)
    - **Validation**: Person-level train/validation/test splits
    
    #### Key Findings
    
    1. **Light smoking** (<10 cigarettes/day) is the strongest predictor of quit success
    2. **High nicotine dependence** (measured by TTFC and CPD) is the primary barrier
    3. **Varenicline and NRT** combined with counseling show highest effectiveness
    4. **Environmental factors** (smokefree homes, household composition) significantly impact outcomes
    5. **Motivation and readiness** are critical timing factors
    
    #### Limitations
    
    - Predictions are **probabilistic estimates**, not guarantees
    - Based on **observational data**, not randomized trials
    - Individual outcomes may vary based on factors not captured in the study
    - **Missing data** (especially CPD: 77% missing) handled via XGBoost native methods
    
    #### Disclaimer
    
    This tool is for **educational and research purposes only**. It is not a substitute 
    for professional medical advice, diagnosis, or treatment. Always consult with a 
    qualified healthcare provider about tobacco cessation.
    
    ---
    
    ### Technical Details
    
    **Model Training**
    - Algorithm: XGBoost 3.1.1
    - Trees: 500
    - Max depth: 4
    - Learning rate: 0.05
    - Class weighting: 2.55 (to address 72% no-quit rate)
    
    **Feature Importance** (Top 5)
    1. CPD Light (<10/day): 20.0% gain
    2. High Dependence: 18.4% gain
    3. Time to First Cigarette: 14.1% gain
    4. CPD Heavy (20+/day): 11.2% gain
    5. Cigarettes Per Day: 4.4% gain
    
    ---
    
    ### Contact & Source Code
    
    - **Repository**: [GitHub - smoking_cessation_ml](https://github.com/Lactoseandtolerance/smoking_cessation_ml)
    - **PATH Study**: [pathstudyinfo.nih.gov](https://pathstudyinfo.nih.gov)
    - **License**: Educational/Research Use
    
    ---
    
    *Last Updated: November 15, 2025*
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Quick Facts")
st.sidebar.info("""
**Study Period**: 2013-2020  
**Sample Size**: 24,576 adults  
**Quit Attempts**: 59,984  
**Model Accuracy**: 87% AUC
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*Data: PATH Study Waves 1-7*")
