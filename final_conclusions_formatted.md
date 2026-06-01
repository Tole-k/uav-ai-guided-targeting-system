# Conclusions

## Timeline of the Attempted Approaches

### 1. Vanilla YOLOv11 (Horizontal Bounding Boxes)
The baseline approach utilized a standard YOLOv11 model with horizontal bounding boxes (where the edges of the segmentation bounding boxes are perpendicular to the borders of the image).

#### Key Takeaways
 The generated plots for this initial training run were unfortunately lost, but key top-level metrics were recovered.
* **Performance Discrepancy:** * **Precision:** 0.86025
  * **Recall:** 0.69043
* **Analysis:** A significant gap existed between precision and recall, indicating that the baseline model suffered from a high rate of false negatives (failing to detect existing objects initially).

---

### 2. YOLOv11 with Oriented Bounding Boxes (OBB)
To better align with the natural geometry and rotation of the target objects, the model was upgraded to use Oriented Bounding Boxes (OBB / oblique bounding boxes).

![F1 Scores of the Basic YOLO OBB](./val/BoxF1_curve.png)
![Confusion Matrix of the Basic YOLO OBB](./val/confusion_matrix.png)
![Normalized Confusion Matrix of the Basic YOLO OBB](./val/confusion_matrix_normalized.png)

#### Key Takeaways
* **Bridged the Performance Gap:** As expected, OBB significantly improved overall performance and successfully narrowed the previously encountered gap between precision and recall.
* **The Background Class Problem:** The primary challenge shifted to the **Background class** (regions containing no objects), which suffered from mutual misclassifications:
  * **False Positives:** Background noise was frequently misclassified as an active class such as `Person`, `Car`, or `Bicycle`.
  * **False Negatives:** Actual objects (specifically `Car` and `Person`) were completely missed and categorized as background.
* **Severe Class Imbalance:** Data analysis revealed a massive disparity between classes within the dataset. For example:
  * **Majority Class (`Person`):** 12,312 instances
  * **Minority Class (`OtherVehicle`):** 148 instances

---

### 3. RT-DETR (Real-Time End-to-End Object Detection)
We evaluated **RT-DETR**, a state-of-the-art vision transformer-based object detection and segmentation model, to see if its attention mechanisms would outperform YOLO's CNN-based architecture.

![F1 Scores of RT-DETR](./rf-detr/val/val/BoxF1_curve.png)
![Confusion Matrix of RT-DETR](./rf-detr/val/val/confusion_matrix.png)
![Normalized Confusion Matrix of RT-DETR](./rf-detr/val/val/confusion_matrix_normalized.png)

#### Key Takeaways
* **Subpar Performance:** RT-DETR performed consistently worse than the default YOLOv11 model across almost all evaluated metrics. even though for majority classes the confusion matrix looks better , over $95%$, but  there is a more visible collapse on the underrepresented classes  -`Dont care object`
* **Analysis:** The SOTA claims of this transformer architecture did not translate to our specific dataset. This suggests that the problem domain or dataset characteristics do not align well with RT-DETR's specific inductive biases.

The 1st observation propelled us to tinker with the loss - and rebalance the quality between the classes!

---

### 4. Loss Weighting Experiments
To combat the severe class imbalance I mentioned earlier, we attempted to modified loss function - added weighting

#### Naive Inverse-Frequency Weighting
* **Result:** Total optimization collapse. Performance severely degraded.
* **Analysis:** Because the minority classes were penalized heavily for
 false positives, the model stopped predicting them entirely to minimize loss. Furthermore, the misclassification issues were not strictly tied to class scarcity; the background class was the worst offender for errors, which inverse-frequency weighting failed to address.
 :disappointed:

#### Effective Number of Samples Weighting
We implemented a smoother weighting scheme based on Class-Balanced Loss to damp the extreme ratios:

$$\text{Weight} = \frac{1 - \beta}{1 - \beta^n}$$

Where $\beta$ was set to $0.999$. 
* **Result:** This approach assigned a weight of `0.0001` to the most numerous class (`Person`) and `0.0007` to the least numerous (`OtherVehicle`) (calculated by me). Even though this damped the penalty (making the minority class only 7 times more heavily weighted, rather than the extreme ratio of inverse-frequency), the practical reality check yielded the same result as before: a complete training collapse.
:confused:
---

### 5. Final Approach: YOLOv11 with Varifocal Loss (VFL)
To resolve both the background misclasificaiton( shapes) and the severe class imbalance simultaneously, we implemented **Varifocal Loss (VFL)**.


![verifocal loss](./uav-ai-guided-targeting-system/images/verifocalLoss.png)

> **How Varifocal Loss Works:**
> Varifocal Loss trains dense object detectors to learn an **IoU-Aware Classification Score (IACS)**. Inspired by Focal Loss, it introduces an **asymmetric weighting** scheme. For positive training examples, it scales the loss based on the target Intersection over Union (IoU), forcing the network to prioritize high-quality, precisely localized bounding boxes. For negative training examples, it uses focal-style down-weighting to heavily suppress background noise, keeping the model from being overwhelmed by non-object regions.

![SOTA Approach F1 Score](./sota-verifocal/phase1-verifocal/BoxF1_curve.png)
![SOTA Approach Confusion Matrix](./sota-verifocal/phase1-verifocal/confusion_matrix.png)
![SOTA Approach Normalized Confusion Matrix](./sota-verifocal/phase1-verifocal/confusion_matrix_normalized.png)

#### Key Takeaways
* **Significant Metric Boost:** The overall F1-confidence score across all classes jumped from our previous best of **0.76 up to 0.83**.
:relieved:
* **Background Error Mitigation:** The confusion matrix shows a drastic reduction in false negatives against the background class, meaning the network is much better at identifying valid segmentation pixels rather than giving up and predicting background.
:smiley:
* **Conclusion:** This asymmetric approach is by far the most successful strategy tested.

---

