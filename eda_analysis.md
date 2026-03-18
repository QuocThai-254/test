Data Strategy & Problem Analysis
Based on the requirements in 
take_home_note.md
 and the findings in 
EDA_Data_Analysis.ipynb
, here is a breakdown of the critical cases and how to solve them within the 3-day constraint.

1. Case: Data Leakage (Train in Test)
Problem: 3 images are present in both data_train and data_test.

Effect: Artificially high test accuracy. The model "remembers" these images rather than generalizing.
Solution: Immediately remove these specific images from the Training set. The test set must remain untouched to serve as a valid "blind" evaluation.
2. Case: Duplicate Images with Conflicting Labels
Problem: Several images are identical (MD5 hash match) but reside in different folders (e.g., other vs ngày_tết_verified).

Effect: The model receives contradictory signals (same input, different expected output), causing high loss and confusion.
Solution:
Rule-based Deduplication: Specific tags should always take priority over the other category.
If an image is in ngày_tết_verified and other, keep it in ngày_tết_verified and delete it from other.
If two specific tags conflict (e.g., tụ_họp vs ngày_tết), manually review or choose the most specific/descriptive one.
3. Case: Extremely Small Dataset (~260 images)
Problem: Deep learning models usually require thousands of images. 260 is very low for 6 classes.

Solution (Non-LLM/Multi-modal):
Transfer Learning (Required): Use a pre-trained backbone (MobileNetV2, ResNet, or EfficientNet) and only train the classifier head.
Heavy Augmentation: Use RandomResizedCrop, ColorJitter, and RandomHorizontalFlip to create "new" training samples.
Class Weighting: Since other has 66 images and tụ_họp only has 24, use WeightedRandomSampler in PyTorch or class_weight in the Loss function to prevent the model from ignoring minority classes.
4. Case: The "Other" Category Imbalance
Problem: other is the largest class and likely contains visually diverse images that don't fit elsewhere.

Solution: Instead of a standard Softmax, consider if some images actually belong to multiple classes. However, for a simple folder-based classifier, the best approach is to ensure the other class is strictly "anything that is NOT one of the 5 specific tags." Relabel images that clearly fit a specific tag and move them out of other.
Recommended 3-Day Timeline
Day 1 (Current): Data Cleaning based on EDA findings (De-duplicate, fix leakage).
Day 2: Training with Transfer Learning + Hyperparameter tuning (Learning rate, Augmentation strength).
Day 3: Evaluation, Error Analysis (Confusion Matrix), and writing the final report.
