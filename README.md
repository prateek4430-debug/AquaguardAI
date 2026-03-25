# AquaGuard AI — Campus Water Quality & Health Risk Predictor

A Python-based machine learning project that checks water safety at multiple points across a college campus. The system reads eight water quality parameters, runs them through a custom-built decision tree, and tells you whether the water is safe, needs attention, or is dangerous — along with specific recommendations for each case.

---

## Why I Built This

Water quality on campuses rarely gets tracked in real time. Most hostel and canteen taps go unchecked for weeks. This project simulates what a smart monitoring system would look like — one that flags unsafe water early and tells the maintenance team exactly what is wrong and what to fix.

No paid libraries. No external ML frameworks. Just Python, NumPy, and Pandas — with the classification logic written from scratch.

---

## What It Does

- Generates a dataset of 500 water samples based on real WHO drinking water guidelines
- Trains a decision tree classifier built entirely without sklearn
- Scans five campus water points and predicts risk level for each
- Shows a confidence breakdown for every prediction
- Gives specific, actionable advice depending on what parameters are off

---

## Parameters Checked

| Parameter | Safe Range | Why It Matters |
|---|---|---|
| pH | 6.5 – 8.5 | Outside this range causes corrosion or scaling in pipes |
| Turbidity | Below 5 NTU | High cloudiness often means sediment or bacteria |
| TDS | Below 600 mg/L | Very high dissolved solids affect taste and health |
| Hardness | Below 300 mg/L | Extremely hard water causes kidney stress over time |
| Chlorine | 0.2 – 1.2 mg/L | Too little means no disinfection; too much is toxic |
| Nitrates | Below 45 mg/L | High nitrates are especially dangerous for children |
| Temperature | 10 – 40 °C | Warm stagnant water accelerates bacterial growth |
| Coliform | Not Detected | Presence indicates sewage contamination |

---

## Project Structure

```
aquaguard-ai/
│
├── aquaguard_ai.py          # Main file — all logic lives here
└── README.md
```

---

## How to Run

**Step 1 — Make sure you have the dependencies**

```bash
pip install pandas numpy
```

**Step 2 — Run the main file**

```bash
python aquaguard_ai.py
```

That is it. The script will:
1. Generate the dataset and save it as a CSV
2. Train the decision tree and print per-class accuracy
3. Scan all five campus water points and print a full report for each
4. Show a campus-wide water safety index at the end

---

## Sample Output

```
  [ AquaGuard AI ] Campus Water Quality Predictor

  Dataset ready — 500 samples saved to water_quality_data.csv

  Building decision tree classifier from scratch...

  Category          Right   Out Of    Score
  ------------------------------------------
  High Risk            12       14    85.7%
  Moderate Risk        43       46    93.5%
  Safe                 38       40    95.0%

  Training complete. Accuracy on test set: 93.00%

  Scanning all registered water points on campus...

====================================================
   AquaGuard AI  |  Campus Water Safety Report
====================================================
   Tap / Point  : Science Lab Water Outlet
   pH           : 6.20
   Turbidity    : 7.80 NTU
   TDS          : 650 mg/L
   Coliform     : PRESENT
----------------------------------------------------
   Result       : HIGH RISK
----------------------------------------------------
   Likelihood breakdown:
     Safe            □□□□□□□□□□□□□□□□□□□□   0%
     Moderate Risk   □□□□□□□□□□□□□□□□□□□□   0%
     High Risk       ■■■■■■■■■■■■■■■■■■■■ 100%
----------------------------------------------------
   What to do:
     1. !! UNSAFE — Do not use this water for drinking or cooking !!
     2. Bacterial contamination found. Flush and chlorinate the line.
     3. Nitrates at 50 mg/L — above WHO 45 mg/L limit.
     4. TDS at 650 mg/L — install RO filter.
     5. Seal this outlet and report to campus maintenance immediately.
====================================================
```

---

## How the Classifier Works

There is no sklearn here. The decision tree is built from scratch using three ideas:

**Gini Impurity** — measures how mixed a group of labels is. A group with all "Safe" labels has zero impurity. A group split equally between all three classes has the highest impurity.

**Best Split Search** — for every feature and every possible threshold, the algorithm tests how well that cut separates the classes. It picks whichever split reduces impurity the most.

**Recursive Tree Growth** — the tree keeps splitting until either all samples in a node belong to the same class, the node has fewer than 10 samples, or we have gone 5 levels deep. At that point it assigns whatever label is most common in that node.

---

## Risk Scoring Logic

Each sample gets a danger score based on how many WHO limits it breaks:

- Coliform detected → +3 points (most serious)
- pH out of range → +2 points
- Turbidity too high → +2 points
- Nitrates too high → +2 points
- TDS too high → +1 point
- Chlorine out of range → +1 point
- Hardness too high → +1 point

Score 0–1 = **Safe** | Score 2–4 = **Moderate Risk** | Score 5+ = **High Risk**

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| NumPy | Any recent | Dataset generation and array operations |
| Pandas | Any recent | DataFrame handling and CSV export |

No sklearn. No scipy. No other ML library.

---

## Possible Extensions

- Connect to actual IoT sensors via serial port or MQTT
- Add a simple web dashboard using Flask or Streamlit
- Store historical readings in SQLite and track water quality trends over time
- Send WhatsApp or email alerts automatically when a tap crosses the danger threshold
- Export weekly PDF reports for campus administration

---

## References

- WHO Guidelines for Drinking-water Quality, 4th Edition — [who.int](https://www.who.int/publications/i/item/9789241549950)
- Bureau of Indian Standards IS 10500:2012 — Drinking Water Specification
- Decision Tree from Scratch — concept based on Breiman et al., *Classification and Regression Trees* (1984)

---

## Author

Built as part of a campus AI project exploring how machine learning can be applied to public health monitoring in educational institutions.

