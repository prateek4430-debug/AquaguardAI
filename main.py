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
def classify_row(node, row_data):
    if node["is_leaf"]:
        return node["decision"]
    if row_data[node["feature"]] < node["threshold"]:
        return classify_row(node["branch_left"], row_data)
    return classify_row(node["branch_right"], row_data)

def train_and_evaluate(dataframe):
    input_cols = [col for col in dataframe.columns if col != "Risk_Level"]
    X_all = dataframe[input_cols].values
    y_all = list(dataframe["Risk_Level"])

    cutoff   = int(0.8 * len(X_all))
    X_tr, X_te = X_all[:cutoff], X_all[cutoff:]
    y_tr, y_te = y_all[:cutoff], y_all[cutoff:]

    trained_tree = grow_tree(X_tr, y_tr)
    predictions  = [classify_row(trained_tree, r) for r in X_te]

    hits = sum(1 for actual, pred in zip(y_te, predictions) if actual == pred)
    overall_acc = hits / len(y_te)

    print(f"\n  {'Category':<16} {'Right':>6} {'Out Of':>8} {'Score':>8}")
    print("  " + "-"*42)
    for cat in sorted(set(y_te)):
        cat_total   = y_te.count(cat)
        cat_correct = sum(1 for a, p in zip(y_te, predictions) if a == p == cat)
        cat_acc     = cat_correct / max(cat_total, 1) * 100
        print(f"  {cat:<16} {cat_correct:>6} {cat_total:>8} {cat_acc:>7.1f}%")

    return trained_tree, overall_acc, input_cols

def compute_danger_score(reading):
    pts = 0
    if reading[0] < WHO_PH_LOW or reading[0] > WHO_PH_HIGH: pts += 2
    if reading[1] > WHO_TURBIDITY:  pts += 2
    if reading[2] > WHO_TDS:        pts += 1
    if reading[5] > WHO_NITRATE:    pts += 2
    if reading[4] < WHO_CHLORINE_LOW or reading[4] > WHO_CHLORINE_HIGH: pts += 1
    if reading[7] == 1:             pts += 3
    if reading[3] > WHO_HARDNESS:   pts += 1
    return pts

def run_prediction(tree, reading):
    outcome = classify_row(tree, reading)
    danger  = compute_danger_score(reading)
    risk_ratio = round(min(danger / 12, 1.0), 2)
    safe_ratio = round(max(1.0 - risk_ratio - 0.1, 0.0), 2)
    mid_ratio  = round(1.0 - risk_ratio - safe_ratio, 2)
    confidence = {"Safe": safe_ratio, "Moderate Risk": mid_ratio, "High Risk": risk_ratio}
    return outcome, confidence
