import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import random

# ==========================================
# 1. CORE FUNCTIONS (CRD ADAPTED)
# ==========================================

def apply_transformation(data, method):
    """Applies the selected transformation."""
    if method == "Square Root (√x + 0.5)":
        return np.sqrt(data + 0.5)
    elif method == "Arcsine (Predicted %)":
        clipped = np.clip(data, 0, 100)
        return np.degrees(np.arcsin(np.sqrt(clipped / 100)))
    else:
        return data

def crd_anova_from_transformed(df_trans, n_treat, n_rep):
    """
    Calculates ANOVA stats specifically for CRD (One-Way ANOVA).
    """
    vals = df_trans.values
    grand_mean = vals.mean()
    N = n_treat * n_rep # Total observations
    
    # Correction Factor
    CF = (vals.sum() ** 2) / N
    
    # Total Sum of Squares
    TSS = (vals ** 2).sum() - CF
    
    # Treatment Sum of Squares
    TrSS = ((vals.sum(axis=1) ** 2).sum() / n_rep) - CF
    
    # Error Sum of Squares (In CRD, Error = Total - Treatment)
    ESS = TSS - TrSS
    
    # Degrees of Freedom
    df_tr = n_treat - 1
    df_total = N - 1
    df_err = df_total - df_tr 
    
    # Mean Squares
    MS_tr = TrSS / df_tr if df_tr > 0 else 0
    MS_err = ESS / df_err if df_err > 0 else 0
    
    # F Statistic
    F_calc = MS_tr / MS_err if MS_err > 0 else 0
    
    if MS_err > 0 and df_err > 0:
        p_val = 1 - stats.f.cdf(F_calc, df_tr, df_err)
        sem = np.sqrt(MS_err / n_rep)
        t_crit = stats.t.ppf(1 - 0.05 / 2, df_err)
        cd = sem * np.sqrt(2) * t_crit
        cv = (np.sqrt(MS_err) / grand_mean) * 100
    else:
        p_val = 1.0; sem = 0; cd = 0; cv = 0

    return {
        "cv": cv, "p_val": p_val, "f_calc": F_calc, 
        "sem": sem, "cd": cd, "ms_err": MS_err,
        "df_tr": df_tr, "df_err": df_err, "df_total": df_total,
        "ss_tr": TrSS, "ss_err": ESS, "ss_total": TSS,
        "ms_tr": MS_tr
    }

def generate_integer_plants(target_mean, n_plants, variation=0.2):
    """
    Generates 'n_plants' INTEGERS that sum to 'round(target_mean * n_plants)'.
    """
    target_sum = int(round(target_mean * n_plants))
    
    # Initial guess
    base_val = target_sum // n_plants
    remainder = target_sum % n_plants
    
    plants = np.full(n_plants, base_val)
    
    if remainder > 0:
        indices = np.random.choice(n_plants, remainder, replace=False)
        plants[indices] += 1
        
    # Shuffle values to add noise
    n_swaps = int(target_sum * variation) 
    
    for _ in range(n_swaps):
        idx1, idx2 = np.random.choice(n_plants, 2, replace=False)
        if plants[idx2] > 0:
            plants[idx1] += 1
            plants[idx2] -= 1
            
    return plants

def generate_exact_mean_data_crd(target_means, n_rep, n_plants, target_cv, transform_type):
    """
    Generates data hierarchy.
    """
    n_treat = len(target_means)
    best_data = None
    best_diff = float('inf')
    
    low_noise = 0.001
    high_noise = np.mean(target_means) * 2.0 
    
    for _ in range(80): # Optimization Loop
        current_noise = (low_noise + high_noise) / 2
        
        final_rep_means = []  
        all_plant_rows = [] 
        
        for t_idx, mean in enumerate(target_means):
            # Generate raw replication values based on normal distribution around mean
            raw_reps = np.random.normal(mean, current_noise, n_rep)
            raw_reps = np.maximum(raw_reps, 0)
            
            actual_reps_for_this_treatment = []
            
            for r_idx, r_target in enumerate(raw_reps):
                # Generate Integers
                p_vals = generate_integer_plants(r_target, n_plants)
                actual_rep_mean = np.mean(p_vals)
                actual_reps_for_this_treatment.append(actual_rep_mean)
                
                # Store plant data
                for p_idx, p_val in enumerate(p_vals):
                    all_plant_rows.append({
                        "Treatment": f"T{t_idx+1}",
                        "Replication": f"R{r_idx+1}",
                        "Plant_No": f"P{p_idx+1}",
                        "Insect_Count": int(p_val)
                    })
            
            final_rep_means.append(actual_reps_for_this_treatment)

        raw_arr = np.array(final_rep_means)
        
        # Check Stats using CRD Logic
        trans_arr = apply_transformation(raw_arr, transform_type)
        temp_df = pd.DataFrame(trans_arr)
        
        # CALL CRD FUNCTION HERE
        stats_res = crd_anova_from_transformed(temp_df, n_treat, n_rep)
        calc_cv = stats_res['cv']
        
        if calc_cv < target_cv:
            low_noise = current_noise 
        else:
            high_noise = current_noise 
            
        cv_diff = abs(calc_cv - target_cv)
        
        if cv_diff < best_diff:
            best_diff = cv_diff
            best_data = (raw_arr, trans_arr, stats_res, pd.DataFrame(all_plant_rows))

    return best_data

# ==========================================
# 2. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="CRD Generator", layout="wide")

st.title("🐞 CRD Integer Generator (Supports Large Plot)")
st.markdown("""
Generates insect counts for **Completely Randomized Design (CRD)**.
* **Standard CRD:** Set Replications > 1.
* **Large Plot Technique:** Set **Replication = 1** and specify Plants per Replication. The tool will treat the plants as the replicates for the ANOVA.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Experimental Design")
# Changed min_value to 1 to allow Large Plot technique
n_reps = st.sidebar.number_input("Number of Replications", min_value=1, value=4)
n_plants = st.sidebar.number_input("Plants per Replication", min_value=1, value=3)

st.sidebar.header("2. Input Means")
st.sidebar.info("Paste Treatment Means (comma separated)")
means_input = st.sidebar.text_area("Treatment Means", "10, 12, 14, 13, 15, 11")

st.sidebar.header("3. Targets")
transform_type = st.sidebar.selectbox("Transformation", ["Square Root (√x + 0.5)", "Arcsine (Predicted %)"])
target_cv = st.sidebar.number_input("Target CV %", min_value=0.1, value=8.5, step=0.1)
sig_req = st.sidebar.selectbox("Significance", ["Significant (S)", "Non-Significant (NS)"])

generate_btn = st.sidebar.button("Generate CRD Data", type="primary")

# --- EXECUTION ---

if generate_btn:
    try:
        target_means = [float(x.strip()) for x in means_input.split(',') if x.strip()]
        if len(target_means) < 3:
            st.error("Enter at least 3 means.")
            st.stop()
    except:
        st.error("Invalid means input.")
        st.stop()
        
    # LOGIC SWITCH FOR LARGE PLOT (1 REP)
    is_large_plot = (n_reps == 1)
    
    if is_large_plot and n_plants < 2:
        st.error("For Single Replication (Large Plot), you must have at least 2 Plants to calculate ANOVA.")
        st.stop()

    with st.spinner("Generating CRD data..."):
        if is_large_plot:
            # Trick the generator: Treat 'Plants' as 'Reps' to get the correct ANOVA variance
            # We generate 'n_plants' reps, each having 1 plant.
            result = generate_exact_mean_data_crd(
                target_means, n_rep=n_plants, n_plants=1, target_cv=target_cv, transform_type=transform_type
            )
        else:
            # Standard Mode
            result = generate_exact_mean_data_crd(
                target_means, n_rep=n_reps, n_plants=n_plants, target_cv=target_cv, transform_type=transform_type
            )

    if result:
        raw_arr, trans_arr, res, df_plants = result
        
        # --- DATA FORMATTING & RENAMING ---
        if is_large_plot:
            # The generator returned 'Replication' as R1..Rn and 'Plant_No' as P1.
            # We need to swap this visually for the user.
            # Old 'Replication' (R1, R2...) -> becomes 'Plant_No' (P1, P2...)
            # Old 'Plant_No' -> Discard, replace with 'RI'
            
            df_plants['Plant_No'] = df_plants['Replication'].str.replace('R', 'P')
            df_plants['Replication'] = 'RI' # Fixed single rep
            
            # For Wide Format: Rows=Treatment, Cols=Plants
            df_wide = df_plants.pivot(index='Treatment', columns='Plant_No', values='Insect_Count')
            
            # Sort columns numerically (P1, P2, P10...)
            cols = sorted(df_wide.columns, key=lambda x: int(x[1:]))
            df_wide = df_wide[cols]
            
        else:
            # Standard CRD Format
            # Pivot: Index=[Treatment, Plant], Columns=Replication
            df_wide = df_plants.pivot(index=['Treatment', 'Plant_No'], columns='Replication', values='Insect_Count')
            
            # Rename columns
            roman_map =
