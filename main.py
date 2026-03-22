import pandas as pd
import numpy as np

SEED = 7
TOTAL_SAMPLES = 500
np.random.seed(SEED)

WHO_PH_LOW = 6.5
WHO_PH_HIGH = 8.5
WHO_TURBIDITY = 5.0
WHO_TDS = 600
WHO_NITRATE = 45
WHO_CHLORINE_LOW = 0.2
WHO_CHLORINE_HIGH = 1.2
WHO_HARDNESS = 300

def make_water_data(size):
    ph_vals      = np.random.normal(7.0, 0.8, size).clip(4.5, 9.5)
    turb_vals    = np.random.exponential(3, size).clip(0.1, 20)
    tds_vals     = np.random.normal(350, 120, size).clip(50, 900)
    hard_vals    = np.random.normal(180, 60, size).clip(50, 400)
    chlor_vals   = np.random.uniform(0.1, 1.5, size)
    nitrate_vals = np.random.exponential(15, size).clip(1, 80)
    temp_vals    = np.random.normal(25, 5, size).clip(10, 40)
    bact_vals    = np.random.choice([0, 1], size, p=[0.75, 0.25])

    labels = []
    for idx in range(size):
        danger = 0
        if ph_vals[idx] < WHO_PH_LOW or ph_vals[idx] > WHO_PH_HIGH: danger += 2
        if turb_vals[idx] > WHO_TURBIDITY:  danger += 2
        if tds_vals[idx] > WHO_TDS:         danger += 1
        if nitrate_vals[idx] > WHO_NITRATE: danger += 2
        if chlor_vals[idx] < WHO_CHLORINE_LOW or chlor_vals[idx] > WHO_CHLORINE_HIGH: danger += 1
        if bact_vals[idx] == 1:             danger += 3
        if hard_vals[idx] > WHO_HARDNESS:   danger += 1

        if danger <= 1:   labels.append("Safe")
        elif danger <= 4: labels.append("Moderate Risk")
        else:             labels.append("High Risk")

    return pd.DataFrame({
        "pH": ph_vals, "Turbidity": turb_vals, "TDS": tds_vals,
        "Hardness": hard_vals, "Chlorine": chlor_vals,
        "Nitrates": nitrate_vals, "Temperature": temp_vals,
        "Coliform": bact_vals, "Risk_Level": labels
    })

def calc_impurity(buckets, all_classes):
    n_total = sum(len(b) for b in buckets)
    impurity = 0.0
    for bucket in buckets:
        bucket_size = len(bucket)
        if bucket_size == 0:
            continue
        squared_sum = sum((bucket.count(c) / bucket_size) ** 2 for c in all_classes)
        impurity += (1.0 - squared_sum) * (bucket_size / n_total)
    return impurity

def find_best_cut(X_data, y_data):
    unique_classes = list(set(y_data))
    top_col, top_val, top_impurity, top_split = None, None, float("inf"), None

    for feature_col in range(X_data.shape[1]):
        thresholds = np.unique(X_data[:, feature_col])
        for threshold in thresholds:
            go_left  = X_data[:, feature_col] < threshold
            go_right = ~go_left
            left_labels  = [y_data[i] for i in range(len(y_data)) if go_left[i]]
            right_labels = [y_data[i] for i in range(len(y_data)) if go_right[i]]
            imp = calc_impurity([left_labels, right_labels], unique_classes)
            if imp < top_impurity:
                top_impurity = imp
                top_col, top_val = feature_col, threshold
                top_split = (go_left, go_right)

    return top_col, top_val, top_split

def dominant_label(label_list):
    return max(set(label_list), key=label_list.count)

def grow_tree(X_data, y_data, current_depth=0, depth_limit=5, leaf_min=10):
    pure      = len(set(y_data)) == 1
    too_small = len(y_data) <= leaf_min
    too_deep  = current_depth >= depth_limit

    if pure or too_small or too_deep:
        return {"is_leaf": True, "decision": dominant_label(list(y_data))}

    cut_col, cut_val, split = find_best_cut(X_data, y_data)

    if split is None:
        return {"is_leaf": True, "decision": dominant_label(list(y_data))}

    go_left, go_right = split
    left_y  = [y_data[i] for i in range(len(y_data)) if go_left[i]]
    right_y = [y_data[i] for i in range(len(y_data)) if go_right[i]]

    if not left_y or not right_y:
        return {"is_leaf": True, "decision": dominant_label(list(y_data))}

    return {
        "is_leaf": False,
        "feature": cut_col,
        "threshold": cut_val,
        "branch_left":  grow_tree(X_data[go_left],  left_y,  current_depth+1, depth_limit, leaf_min),
        "branch_right": grow_tree(X_data[go_right], right_y, current_depth+1, depth_limit, leaf_min)
    }
