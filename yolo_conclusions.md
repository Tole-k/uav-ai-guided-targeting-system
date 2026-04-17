# YOLO Model Performance & Recall Improvement Strategy (Infrared)


## Tables 
| Metric | Value |
|--------|------|
| train/box_loss | 0.87137 |
| train/cls_loss | 0.28845 |
| train/dfl_loss | 0.00156 |
| metrics/precision (B) | 0.86025 |
| metrics/recall (B) | 0.69043 |
| metrics/mAP50 (B) | 0.78627 |
| metrics/mAP50-95 (B) | 0.53438 |
| lr/pg0 | 0.000286083 |
| lr/pg1 | 0.000286083 |
| lr/pg2 | 0.000286083 |


| Metric | Value |
|--------|------|
| val/box_loss | 1.37234 |
| val/cls_loss | 0.57208 |
| val/dfl_loss | 0.00321 |



### Losses
- **box_loss**: Measures how well predicted bounding boxes match ground truth. Lower = better localization.
- **cls_loss**: Classification error for predicted object classes. Lower = better classification.
- **dfl_loss (Distribution Focal Loss)**: Refines bounding box edges; improves localization precision. Very small values are normal.

### Detection Metrics
- **Precision**: Fraction of predicted objects that are correct. High precision = few false positives.
- **Recall**: Fraction of real objects that were detected. Lower recall means missed detections.
- **mAP50**: Mean Average Precision at IoU = 0.5. Measures detection quality under a lenient threshold.
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95. Stricter and more representative of overall performance.

### Validation vs Training
- Higher validation loss compared to training loss.
- A gap between precision and recall suggests imbalance: high precision + lower recall → model is conservative.

### Learning Rate
- **lr/pg0, pg1, pg2**: Learning rates for different parameter groups. Stable values indicate steady training phase.



## Summary of Current Performance
The model shows **strong precision but relatively weak recall**, meaning:
- It is **accurate when detecting objects**
- But **misses lots of them**

### Key Metrics
- **Precision:** 0.86 → High 
- **Recall:** 0.69 → Moderate 
- **mAP50:** 0.79 → Good overall detection
- **mAP50-95:** 0.53 → localization accuracy

---

##  Key Conclusion
The model behaves as a:

> **High-precision, conservative detector**



---

##  Boosting Consideration


---

##  possible Strategies (Boosting-Inspired)

###  1. Hard Example Mining 
- Run inference on training/validation data
- Identify:
  - False negatives (missed objects)
  - Low-confidence detections
- Oversample or duplicate these images during training
- check how do the metric looks inside each class 


---

###  Data Augmentation (Infrared-Specific)
Apply augmentations that simulate difficult IR conditions:
- Noise injection (thermal noise)
- Blur
- Contrast/brightness shifts
- Small object scaling


---

###  3. Loss Reweighting 
Adjust training to penalize missed detections more:
- Increase objectness / box loss weights


- Recall increase
- Precision may slightly decrease

---

---

### 4.  Lower Confidence Threshold (Quick Win)
At inference time:
```python
conf = 0.25 → try 0.1–0.15 
for the classifier  -so on the roc curve higher recall lowe precision
possible plotting for some kinds of decision threshold  
```
### 5. Augment with angel and height data via some encoder

- embedd the height and angel via some small mlp and concatenate  with decoder ?



#
# Todo
priority order via numbers 
- 1. check where the model goes wrong (with respect to labels, with respect to time of day) , examination of the failiures
- 3. check the distribution between training and validation
- 4. weighting of the loss w.r.t to the class -> weight more harder cases
- 2. get rid of the DontCare label from the dataset class 