import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import train_model

def main():
    print("=" * 80)
    print("[LOBO Cross-Validation via Aligned Q-PEAK Architecture]")
    print("=" * 80)

    # 1. Load data
    df_raw = pd.read_csv("Battery_RUL.csv")
    
    # 2. Attach battery ID
    raw_with_id = train_model.attach_battery_id(df_raw)
    
    # 3. Pre-extract all features globally to avoid O(N^2) feature engineering bottleneck in each fold
    print("[INFO] Performing global cleaning and feature engineering...")
    clean_params = train_model.fit_cleaning_params(raw_with_id)
    df_clean = train_model.apply_cleaning(raw_with_id, clean_params)
    df_all_feats = train_model.engineer_features(df_clean)
    
    # 4. LOBO validation loop
    batts = np.array(sorted(df_all_feats["Battery_ID"].unique()))
    default_params = train_model.Params()
    
    rmses = []
    
    for test_b in batts:
        # Split features directly
        dtr = df_all_feats[df_all_feats["Battery_ID"] != test_b].reset_index(drop=True)
        dte = df_all_feats[df_all_feats["Battery_ID"] == test_b].reset_index(drop=True)
        
        # Get feature lists
        clock_feats, full_feats, kan_feats, _, _ = train_model.get_feature_sets(df_all_feats)
        baseline_feats = [c for c in train_model.CLOCK_BASELINE_FEATURES if c in dtr.columns]
        residual_feats = full_feats if (train_model.USE_LEGACY_RESIDUAL_BANK or train_model.USE_SINGLE_RESIDUAL_BRANCH or train_model.USE_QGWO_ENSEMBLE_BRANCHES) else []
        
        # Run pipeline
        out = train_model.run_full_pipeline(
            dtr, dte,
            residual_feats=residual_feats,
            kan_feats=kan_feats,
            baseline_feats=baseline_feats,
            seed=45,
            params=default_params,
            fast_mode=True
        )
        
        y_te = out["y_te"]
        cau_raw = out["tracks"]["full_causal_pims"][1]
        
        # Apply calibration
        max_clip = float(np.max(out["y_tr"]) * 1.1)
        post_cal = train_model.fit_train_only_post_calibrator(
            dtr,
            out["y_tr"],
            out["tracks"]["full_causal_pims"][0],
            mode=train_model.POST_CALIBRATION_MODE
        )
        cau, _ = train_model.apply_train_only_post_calibrator(dte, cau_raw, post_cal, max_clip=max_clip)
        
        # Calculate RMSE
        m = train_model.calc_metrics(y_te, cau)
        print(f"LOBO Battery {test_b} RMSE: {m['RMSE']:.4f}")
        rmses.append(m['RMSE'])
        
    print("-" * 80)
    print(f"Mean LOBO RMSE: {np.mean(rmses):.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
