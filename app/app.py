import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import random

# Define functions directly in the file to avoid import issues
def load_data(data_path):
    return pd.read_csv(data_path)

def load_model(model_path):
    return joblib.load(model_path)

def get_random_defaults(data, available_features):
    """Get random default values from the dataset"""
    # Select a random row from the dataset
    random_row = data.sample(n=1).iloc[0]
    defaults = {}
    
    for col in available_features:
        if col in data.columns:
            if data[col].dtype == 'object':
                defaults[col] = random_row[col]
            else:
                # Ensure numeric values are integers for consistency
                defaults[col] = int(random_row[col])
        else:
            # Fallback for missing columns
            if data[col].dtype == 'object':
                defaults[col] = random.choice(data[col].unique().tolist())
            else:
                defaults[col] = int(random.uniform(data[col].min(), data[col].max()))
    
    return defaults

def preprocess_input(input_dict, columns, data):
    # Create a DataFrame with proper column names to maintain feature names
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    # Feature mapping from current dataset to model features
    feature_mapping = {
        'policy_csl': 'csl_per_accident',  # Map policy_csl to csl_per_accident
        'auto_year': 'vehicle_age',        # Map auto_year to vehicle_age
        'insured_zip': 'insured_zip',      # Keep as is
        'policy_number': 'policy_number'   # Keep as is
    }
    
    # Create a new row with mapped features
    mapped_input = {}
    
    # First, use the provided input values
    for col, value in input_dict.items():
        if col in feature_mapping:
            mapped_input[feature_mapping[col]] = value
        else:
            mapped_input[col] = value
    
    # Handle special case for vehicle_age (calculate from auto_year if available)
    if 'auto_year' in input_dict and 'vehicle_age' not in mapped_input:
        current_year = 2024  # Assuming current year
        mapped_input['vehicle_age'] = current_year - float(input_dict['auto_year'])
    
    # Handle special case for CSL fields
    if 'policy_csl' in input_dict:
        csl_value = input_dict['policy_csl']
        if '/' in str(csl_value):
            parts = str(csl_value).split('/')
            mapped_input['csl_per_person'] = float(parts[0])
            mapped_input['csl_per_accident'] = float(parts[1])
        else:
            mapped_input['csl_per_person'] = float(csl_value) if str(csl_value).replace('.','').isdigit() else 0
            mapped_input['csl_per_accident'] = float(csl_value) if str(csl_value).replace('.','').isdigit() else 0
    
    # Handle categorical variables - convert YES/NO to 1/0
    categorical_mappings = {
        'YES': 1, 'NO': 0, 'Y': 1, 'N': 0,
        'MALE': 1, 'FEMALE': 0,
        'True': 1, 'False': 0,
        'true': 1, 'false': 0
    }
    
    # Apply categorical mappings
    for col, value in mapped_input.items():
        if str(value).upper() in categorical_mappings:
            mapped_input[col] = categorical_mappings[str(value).upper()]
        elif isinstance(value, str):
            # For other string values, try to encode them using the original data
            if col in data.columns:
                unique_vals = data[col].unique()
                if value in unique_vals:
                    # Use label encoding based on unique values in training data
                    mapped_input[col] = list(unique_vals).index(value)
                else:
                    mapped_input[col] = 0  # Default value for unseen categories
    
    # Create DataFrame with the expected features
    result_df = pd.DataFrame([mapped_input])
    
    # Ensure all expected columns are present, fill missing with 0
    for col in columns:
        if col not in result_df.columns:
            result_df[col] = 0
    
    # Reorder columns to match model expectations
    result_df = result_df.reindex(columns=columns, fill_value=0)
    
    # Ensure all values are numeric
    for col in result_df.columns:
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)
    
    return result_df

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'insurance_claims.csv')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'extra_trees_best_model.pkl')

def main():
    st.title('🔍 Auto Insurance Fraud Detection')
    st.write('Enter claim details to predict if a claim is fraudulent.')

    # Debug: Show file paths
    st.write(f"Data path: {DATA_PATH}")
    st.write(f"Model path: {MODEL_PATH}")
    st.write(f"Data file exists: {os.path.exists(DATA_PATH)}")
    st.write(f"Model file exists: {os.path.exists(MODEL_PATH)}")

    # Load data and model
    with st.spinner('Loading data and model...'):
        try:
            st.write("Loading data...")
            data = load_data(DATA_PATH)
            st.write("Loading model...")
            model = load_model(MODEL_PATH)
            st.success("Data and model loaded successfully!")
            
            # Get the model's expected feature names
            if hasattr(model, 'feature_names_in_'):
                model_features = list(model.feature_names_in_)
                st.write("Model expects these features:", model_features)
            else:
                # Fallback: assume these are the expected features based on the error
                model_features = [
                    'months_as_customer', 'age', 'policy_state', 'policy_deductable', 
                    'policy_annual_premium', 'umbrella_limit', 'insured_sex', 
                    'insured_education_level', 'insured_occupation', 'insured_hobbies', 
                    'insured_relationship', 'capital-gains', 'capital-loss', 
                    'incident_type', 'collision_type', 'incident_severity', 
                    'authorities_contacted', 'incident_state', 'incident_city', 
                    'incident_location', 'incident_hour_of_the_day', 
                    'number_of_vehicles_involved', 'property_damage', 'bodily_injuries', 
                    'witnesses', 'police_report_available', 'total_claim_amount', 
                    'injury_claim', 'property_claim', 'vehicle_claim', 'auto_make', 
                    'auto_model', 'csl_per_person', 'csl_per_accident', 'vehicle_age'
                ]
        except Exception as e:
            st.error(f"Error loading data or model: {e}")
            return

    # Show data exploration
    with st.expander('📊 Show Dataset'):
        st.dataframe(data.head(100))
        st.write('Shape:', data.shape)
        st.write('Columns:', list(data.columns))

    # Prepare input form
    st.header('📝 Enter Claim Details')
    
    # Add navigation options
    input_mode = st.radio(
        "Choose input method:",
        ["🏃 Quick Form (All at once)", "📋 Step-by-Step"],
        horizontal=True
    )
    
    # Use only the features that are available in the dataset and don't conflict with model
    available_features = [col for col in data.columns if col != 'fraud_reported' and col not in ['_c39', 'policy_bind_date', 'incident_date']]
    
    # Add clear button with session state
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button('🗑️ Clear Values', help="Reset all fields to empty values"):
            # Clear session state for all form fields
            for key in list(st.session_state.keys()):
                if key.startswith('field_'):
                    del st.session_state[key]
            # Set a flag to indicate fields were cleared
            st.session_state['fields_cleared'] = True
            st.session_state['current_step'] = 0
            st.rerun()
    
    with col2:
        if st.button('🎲 New Random', help="Generate new random default values"):
            # Generate new random defaults
            st.session_state.random_defaults = get_random_defaults(data, available_features)
            # Clear existing form fields to force refresh with new defaults
            for key in list(st.session_state.keys()):
                if key.startswith('field_'):
                    del st.session_state[key]
            st.session_state['current_step'] = 0
            st.rerun()
    
    input_dict = {}
    
    # Initialize random defaults for all features
    if 'random_defaults' not in st.session_state or st.session_state.get('fields_cleared', False):
        st.session_state.random_defaults = get_random_defaults(data, available_features)
        if st.session_state.get('fields_cleared', False):
            st.session_state['fields_cleared'] = False  # Reset the cleared flag
    
    # Create mapping for conflicting features
    feature_display_mapping = {
        'policy_csl': 'Policy CSL (will be split into per_person/per_accident)',
        'auto_year': 'Auto Year (will be converted to vehicle_age)',
        'policy_number': 'Policy Number',
        'insured_zip': 'Insured ZIP Code'
    }
    
    if input_mode == "📋 Step-by-Step":
        # Step-by-step input mode
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0
        
        current_step = st.session_state.current_step
        total_steps = len(available_features)
        
        # Progress bar
        progress = (current_step + 1) / total_steps
        st.progress(progress, text=f"Step {current_step + 1} of {total_steps}")
        
        if current_step < total_steps:
            col = available_features[current_step]
            display_name = feature_display_mapping.get(col, col)
            field_key = f'field_{col}'
            
            st.subheader(f"Step {current_step + 1}: {display_name}")
            
            # Check if fields were just cleared
            fields_were_cleared = st.session_state.get('fields_cleared', False)
            
            if data[col].dtype == 'object':
                options = data[col].unique().tolist()
                options_with_empty = ['-- Select --'] + options
                
                # Use random default if fields were cleared or no session state
                if fields_were_cleared or field_key not in st.session_state:
                    # Use random default from session state
                    random_default = st.session_state.random_defaults.get(col)
                    default_index = options_with_empty.index(random_default) if random_default in options_with_empty else 0
                else:
                    default_index = options_with_empty.index(st.session_state[field_key]) if st.session_state[field_key] in options_with_empty else 0
                
                selected_value = st.selectbox(
                    f'{display_name}', 
                    options_with_empty,
                    index=default_index,
                    key=field_key
                )
                
                if selected_value != '-- Select --':
                    input_dict[col] = selected_value
                    
            else:
                min_val = int(data[col].min())
                max_val = int(data[col].max())
                
                # Use random default if fields were cleared or no session state
                if fields_were_cleared or field_key not in st.session_state:
                    random_default = st.session_state.random_defaults.get(col, int(data[col].mean()))
                    # Ensure random_default is an integer
                    random_default = int(random_default)
                    input_dict[col] = st.number_input(
                        f'{display_name}', 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=random_default,
                        step=1,
                        format="%d",
                        key=field_key
                    )
                else:
                    # Use existing session state value
                    session_val = int(st.session_state[field_key])
                    input_dict[col] = st.number_input(
                        f'{display_name}', 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=session_val,
                        step=1,
                        format="%d",
                        key=field_key
                    )
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if current_step > 0:
                    if st.button('⬅️ Previous'):
                        st.session_state.current_step -= 1
                        st.rerun()
            
            with col2:
                if current_step < total_steps - 1:
                    # Check if current field has a value before allowing next
                    can_proceed = col in input_dict and input_dict[col] is not None
                    if st.button('➡️ Next', disabled=not can_proceed):
                        st.session_state.current_step += 1
                        st.rerun()
            
            with col3:
                if current_step == total_steps - 1:
                    if st.button('✅ Finish'):
                        st.session_state.current_step = total_steps
                        st.rerun()
        
        # Collect all previously entered values
        for i, col in enumerate(available_features):
            field_key = f'field_{col}'
            if field_key in st.session_state and st.session_state[field_key] is not None:
                if data[col].dtype == 'object' and st.session_state[field_key] != '-- Select --':
                    input_dict[col] = st.session_state[field_key]
                elif data[col].dtype != 'object':
                    input_dict[col] = st.session_state[field_key]
    
    else:
        # Quick form mode (original behavior)
        for col in available_features:
            display_name = feature_display_mapping.get(col, col)
            field_key = f'field_{col}'
            
            # Check if fields were just cleared
            fields_were_cleared = st.session_state.get('fields_cleared', False)
            
            if data[col].dtype == 'object':
                options = data[col].unique().tolist()
                # Add an empty option at the beginning
                options_with_empty = ['-- Select --'] + options
                
                # Use random default if fields were cleared or no session state
                if fields_were_cleared or field_key not in st.session_state:
                    # Use random default from session state
                    random_default = st.session_state.random_defaults.get(col)
                    default_index = options_with_empty.index(random_default) if random_default in options_with_empty else 0
                else:
                    default_index = options_with_empty.index(st.session_state[field_key]) if st.session_state[field_key] in options_with_empty else 0
                
                selected_value = st.selectbox(
                    f'{display_name}', 
                    options_with_empty,
                    index=default_index,
                    key=field_key
                )
                
                # Only add to input_dict if a valid option is selected
                if selected_value != '-- Select --':
                    input_dict[col] = selected_value
            else:
                min_val = int(data[col].min())
                max_val = int(data[col].max())
                
                # Use random default if fields were cleared or no session state
                if fields_were_cleared or field_key not in st.session_state:
                    # Use random default from session state
                    random_default = st.session_state.random_defaults.get(col, int(data[col].mean()))
                    input_dict[col] = st.number_input(
                        f'{display_name}', 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=random_default,
                        step=1,
                        format="%d",
                        key=field_key
                    )
                else:
                    # Use existing session state value or random default
                    default_val = st.session_state.get(field_key, st.session_state.random_defaults.get(col, int(data[col].mean())))
                    # Ensure default_val is an integer
                    default_val = int(default_val)
                    input_dict[col] = st.number_input(
                        f'{display_name}', 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=default_val,
                        step=1,
                        format="%d",
                        key=field_key
                    )
    
    # Clear the flag after processing
    if 'fields_cleared' in st.session_state:
        del st.session_state['fields_cleared']

    if st.button('🔍 Predict Fraud', type='primary'):
        # Check if all required fields are filled
        missing_fields = []
        for col in available_features:
            if col not in input_dict or input_dict[col] is None:
                missing_fields.append(col)
        
        if missing_fields:
            st.warning(f"Please fill in the following fields: {', '.join(missing_fields[:5])}{'...' if len(missing_fields) > 5 else ''}")
        else:
            try:
                # Debug: Show model feature names if available
                if hasattr(model, 'feature_names_in_'):
                    st.write("Model expects these features:", list(model.feature_names_in_))
                
                X = preprocess_input(input_dict, model_features, data)
                st.write("Preprocessed input shape:", X.shape)
                st.write("Preprocessed input columns:", list(X.columns))
                
                prediction = model.predict(X)[0]
                
                if prediction == "Y" or prediction == 1:
                    st.error('🚨 FRAUD DETECTED')
                else:
                    st.success('✅ NOT FRAUD')
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.write("Available columns in data:", list(data.columns))
                st.write("Expected feature columns:", model_features)
                
                # Debug: Show model feature names if available
                if hasattr(model, 'feature_names_in_'):
                    st.write("Model was trained with these features:", list(model.feature_names_in_))

if __name__ == '__main__':
    main()
