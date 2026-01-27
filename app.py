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
    Differences from RBD: No Block/Rep Sum of Squares.
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
    df_err = df_total - df_tr  # or n_treat * (n_rep - 1)
    
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
    Generates data hierarchy: Treatment -> Replication -> Integer Plants
    Optimizes for CRD ANOVA.
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
            # Center them exactly to the mean to avoid drift, then add noise back effectively via plant generation
            # For CRD, we want the variation BETWEEN reps to equal the noise
            
            # Ensure non-negative
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

st.title("🐞 CRD Integer Plant-Wise Generator")
st.markdown("""
This tool generates whole number insect counts for a **Completely Randomized Design (CRD)**.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Experimental Design")
n_reps = st.sidebar.number_input("Number of Replications", min_value=2, value=4)
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

    with st.spinner("Generating CRD data..."):
        result = generate_exact_mean_data_crd(
            target_means, n_reps, n_plants, target_cv, transform_type
        )

    if result:
        raw_arr, trans_arr, res, df_plants = result
        
        # --- PREPARE EXPORT FORMAT (WIDE) ---
        # 1. Pivot the data: Index=[Treatment, Plant], Columns=Replication
        df_wide = df_plants.pivot(index=['Treatment', 'Plant_No'], columns='Replication', values='Insect_Count')
        
        # 2. Rename columns
        roman_map = {'R1': 'RI', 'R2': 'RII', 'R3': 'RIII', 'R4': 'RIV', 'R5': 'RV', 'R6': 'RVI'}
        new_cols = [roman_map.get(c, c) for c in df_wide.columns]
        df_wide.columns = new_cols

        # --- DISPLAY RESULTS ---
        st.markdown("### 📊 Statistical Summary (CRD)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Achieved CV %", f"{res['cv']:.2f}%", delta=f"{res['cv']-target_cv:.2f}")
        is_sig = res['p_val'] < 0.05
        sig_text = "Significant" if is_sig else "Non-Significant"
        m2.metric("P-Value", f"{res['p_val']:.4f}")
        m3.metric("Significance", sig_text, delta="Match" if (sig_text[0] == sig_req[0]) else "Mismatch")
        m4.metric("SEm / CD", f"{res['sem']:.3f} / {res['cd']:.3f}")
        
        st.divider()

        # --- WIDE FORMAT TAB ---
        t1, t2, t3 = st.tabs(["📥 Export Wide Format (CSV)", "📊 Replication Summary", "📉 ANOVA Table"])
        
        with t1:
            st.subheader("Wide Format Data")
            st.dataframe(df_wide, use_container_width=True)
            
            # CSV Button
            csv_wide = df_wide.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Wide Format CSV",
                data=csv_wide,
                file_name="crd_wide_format_data.csv",
                mime="text/csv",
                type="primary"
            )

        with t2:
            st.write("**Replication Means (Calculated)**")
            cols = [f"R{i+1}" for i in range(n_reps)]
            rows = [f"T{i+1}" for i in range(len(target_means))]
            df_reps = pd.DataFrame(raw_arr, index=rows, columns=cols)
            st.dataframe(df_reps.style.format("{:.2f}"))

        with t3:
            st.write("**ANOVA on Transformed Data (CRD)**")
            anova_data = {
                'SOURCE': ['Treatment', 'Error', 'Total'],
                'DF': [res['df_tr'], res['df_err'], res['df_total']],
                'SS': [res['ss_tr'], res['ss_err'], res['ss_total']],
                'MS': [res['ms_tr'], res['ms_err'], ''],
                'F-Calc': [f"{res['f_calc']:.4f}", '', ''], 
                'Sig?': [sig_text, '', '']
            }
            
            df_disp = pd.DataFrame(anova_data)
            st.table(df_disp)