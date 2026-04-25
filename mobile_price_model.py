import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings 
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from sklearn.metrics import confusion_matrix, classification_report


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')


print(train_df.head())


# 1. Check the shape (number of rows and columns) of both datasets
print("=" * 50)
print("DATASET SHAPES")
print("=" * 50)
print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape:  {test_df.shape}")

# 2. Display the first few rows of the training data
print("\n" + "=" * 50)
print("FIRST 5 ROWS OF TRAINING DATA")
print("=" * 50)
print(train_df.head())

# 3. Display the first few rows of the testing data
print("\n" + "=" * 50)
print("FIRST 5 ROWS OF TESTING DATA")
print("=" * 50)
print(test_df.head())




# ==============================================
# 1. CHECK DATA TYPES OF ALL COLUMNS
# ==============================================
print("=" * 50)
print("DATA TYPES - TRAINING DATA")
print("=" * 50)
print(train_df.dtypes)

print("\n" + "=" * 50)
print("DATA TYPES - TESTING DATA")
print("=" * 50)
print(test_df.dtypes)

# ==============================================
# 2. CHECK FOR MISSING VALUES
# ==============================================
print("\n" + "=" * 50)
print("MISSING VALUES - TRAINING DATA")
print("=" * 50)
print(train_df.isnull().sum())

print("\n" + "=" * 50)
print("MISSING VALUES - TESTING DATA")
print("=" * 50)
print(test_df.isnull().sum())

# ==============================================
# 3. BASIC STATISTICAL SUMMARY
# ==============================================
print("\n" + "=" * 50)
print("STATISTICAL SUMMARY - TRAINING DATA")
print("=" * 50)
print(train_df.describe())

# ==============================================
# 4. TARGET VARIABLE DISTRIBUTION
# ==============================================
print("\n" + "=" * 50)
print("TARGET VARIABLE (price_range) DISTRIBUTION")
print("=" * 50)
print(train_df['price_range'].value_counts().sort_index())
print(f"\nPercentage distribution:")
print(train_df['price_range'].value_counts(normalize=True).sort_index() * 100)

# ==============================================
# 5. GENERAL INFO ABOUT THE DATAFRAME
# ==============================================
print("\n" + "=" * 50)
print("DATAFRAME INFO - TRAINING DATA")
print("=" * 50)
print(train_df.info())




# # ==============================================
# # PART A: TARGET DISTRIBUTION VISUALIZATION
# # ==============================================

# # Set the style for all plots
# sns.set_style("whitegrid")
# plt.figure(figsize=(10, 6))

# # Create a count plot for price_range
# sns.countplot(data=train_df, x='price_range', palette='viridis')

# # Add title and labels
# plt.title('Distribution of Mobile Price Ranges', fontsize=16, fontweight='bold')
# plt.xlabel('Price Range (0=Low, 1=Medium, 2=High, 3=Very High)', fontsize=12)
# plt.ylabel('Number of Phones', fontsize=12)

# # Add value labels on top of each bar
# for i in range(4):
#     plt.text(i, train_df['price_range'].value_counts().sort_index()[i] + 10, 
#              str(train_df['price_range'].value_counts().sort_index()[i]), 
#              ha='center', fontsize=12, fontweight='bold')

# plt.tight_layout()
# plt.show()






# # ==============================================
# # PART B: CORRELATION HEATMAP
# # ==============================================

# plt.figure(figsize=(14, 10))

# # Calculate correlation matrix
# correlation_matrix = train_df.corr()

# # Create heatmap
# sns.heatmap(correlation_matrix, 
#             annot=True,           # Show correlation values
#             cmap='coolwarm',      # Color scheme (red = positive, blue = negative)
#             center=0,             # Center the colormap at 0
#             fmt='.2f',            # Format numbers to 2 decimal places
#             square=True,          # Make cells square
#             linewidths=0.5,       # Add lines between cells
#             cbar_kws={"shrink": 0.8})  # Shrink colorbar

# plt.title('Feature Correlation Heatmap', fontsize=18, fontweight='bold')
# plt.tight_layout()
# plt.show()



# # ==============================================
# # PART B-2: CORRELATION WITH TARGET VARIABLE
# # ==============================================

# # Get correlations with price_range only
# price_corr = correlation_matrix['price_range'].drop('price_range').sort_values(ascending=False)

# print("=" * 50)
# print("FEATURE CORRELATION WITH PRICE_RANGE")
# print("=" * 50)
# print(price_corr)

# # Visualize top correlations
# plt.figure(figsize=(10, 8))
# price_corr.plot(kind='barh', color='steelblue')
# plt.title('Correlation of Features with Price Range', fontsize=16, fontweight='bold')
# plt.xlabel('Correlation Coefficient', fontsize=12)
# plt.ylabel('Features', fontsize=12)
# plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
# plt.tight_layout()
# plt.show()


# # ==============================================
# # PART C: BOX PLOTS - FEATURES VS PRICE_RANGE
# # ==============================================

# # Select features that likely have strong relationships with price
# features_to_plot = ['ram', 'battery_power', 'px_height', 'px_width', 'int_memory']

# fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# axes = axes.flatten()  # Flatten to 1D array for easy iteration

# for i, feature in enumerate(features_to_plot):
#     sns.boxplot(data=train_df, x='price_range', y=feature, ax=axes[i], palette='Set2')
#     axes[i].set_title(f'{feature} vs Price Range', fontsize=14, fontweight='bold')
#     axes[i].set_xlabel('Price Range', fontsize=12)
#     axes[i].set_ylabel(feature, fontsize=12)

# # Remove the empty subplot (since we have 5 features but 6 slots)
# if len(features_to_plot) < len(axes):
#     fig.delaxes(axes[-1])

# plt.suptitle('Feature Distributions Across Price Ranges', fontsize=18, fontweight='bold', y=1.02)
# plt.tight_layout()
# plt.show()




# # ==============================================
# # PART D: DISTRIBUTION PLOTS (HISTOGRAMS)
# # ==============================================

# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# axes = axes.flatten()

# key_features = ['ram', 'battery_power', 'clock_speed', 'mobile_wt']

# for i, feature in enumerate(key_features):
#     sns.histplot(data=train_df, x=feature, hue='price_range', 
#                  kde=True, ax=axes[i], palette='viridis', alpha=0.6)
#     axes[i].set_title(f'Distribution of {feature} by Price Range', fontsize=14, fontweight='bold')
#     axes[i].set_xlabel(feature, fontsize=12)
#     axes[i].set_ylabel('Count', fontsize=12)

# plt.suptitle('Feature Distributions Colored by Price Range', fontsize=18, fontweight='bold', y=1.02)
# plt.tight_layout()
# plt.show()





print("=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

# -------------------------------------------------
# 5.1: Separate Features (X) and Target (y) for Training Data
# -------------------------------------------------

# Define target variable name
target_column = 'price_range'

# For training data: X = all columns except target, y = only target
X = train_df.drop(columns=[target_column])
y = train_df[target_column]

print(f"\nTraining Features (X) shape: {X.shape}")
print(f"Training Target (y) shape:   {y.shape}")
print(f"\nFirst 5 values of y:\n{y.head().tolist()}")

# -------------------------------------------------
# 5.2: Prepare Test Data
# -------------------------------------------------

# For test data: Keep 'id' separate, then create X_test from remaining features
test_ids = test_df['id']
X_test = test_df.drop(columns=['id'])

print(f"\nTest IDs shape:{test_ids.shape}")
print(f"Test Features shape: {X_test.shape}")

# -------------------------------------------------
# 5.3: Split Training Data into Train and Validation Sets
# -------------------------------------------------

# Split the data: 80% for training, 20% for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2,        # 20% for validation
    random_state=42,      # For reproducibility
    stratify=y            # Maintain class balance in both splits
)

print(f"\nAfter Train/Validation Split:")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape:   {y_val.shape}")

# Verify class balance in splits
print(f"\nClass distribution in y_train:")
print(y_train.value_counts().sort_index())
print(f"\nClass distribution in y_val:")
print(y_val.value_counts().sort_index())

# -------------------------------------------------
# 5.4: Feature Scaling (Standardization)
# -------------------------------------------------

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on training data and transform training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test data using the SAME scaler
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeature Scaling Complete!")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_val_scaled shape:   {X_val_scaled.shape}")
print(f"X_test_scaled shape:  {X_test_scaled.shape}")

# -------------------------------------------------
# 5.5: Verify Scaling Results
# -------------------------------------------------

# Convert scaled arrays back to DataFrames for easier viewing
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X.columns)

print("\n" + "=" * 50)
print("SCALING VERIFICATION")
print("=" * 50)

print("\nOriginal Training Data - First 3 rows, First 5 columns:")
print(X_train.iloc[:3, :5])

print("\nScaled Training Data - First 3 rows, First 5 columns:")
print(X_train_scaled_df.iloc[:3, :5])

print("\nMean of scaled features (should be close to 0):")
print(X_train_scaled_df.mean().head())

print("\nStandard deviation of scaled features (should be close to 1):")
print(X_train_scaled_df.std().head())




print("=" * 60)
print("MODEL TRAINING AND COMPARISON")
print("=" * 60)

# -------------------------------------------------
# 6.1: Define Models Dictionary
# -------------------------------------------------

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Classifier': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

print("\nModels to be trained:")
for model_name in models.keys():
    print(f"  • {model_name}")

# -------------------------------------------------
# 6.2: Train Each Model and Evaluate
# -------------------------------------------------

# Dictionary to store results
results = {}

print("\n" + "=" * 60)
print("TRAINING PROGRESS")
print("=" * 60)

for model_name, model in models.items():
    
    # Start timer
    start_time = time.time()
    
    # Train the model on scaled training data
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on training data
    y_train_pred = model.predict(X_train_scaled)
    
    # Make predictions on validation data
    y_val_pred = model.predict(X_val_scaled)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Store results
    results[model_name] = {
        'model': model,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'training_time': training_time,
        'y_val_pred': y_val_pred
    }
    
    # Print progress
    print(f"\n{model_name}:")
    print(f"  Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Training Time:       {training_time:.4f} seconds")

# -------------------------------------------------
# 6.3: Model Comparison Summary
# -------------------------------------------------

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

# Create a DataFrame for easy comparison
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [results[m]['train_accuracy'] for m in results],
    'Val Accuracy': [results[m]['val_accuracy'] for m in results],
    'Training Time (s)': [results[m]['training_time'] for m in results]
})

# Calculate overfitting gap (difference between train and validation accuracy)
comparison_df['Overfitting Gap'] = comparison_df['Train Accuracy'] - comparison_df['Val Accuracy']

# Sort by validation accuracy (descending)
comparison_df = comparison_df.sort_values('Val Accuracy', ascending=False).reset_index(drop=True)

print("\n")
print(comparison_df.to_string(index=True))

# -------------------------------------------------
# 6.4: Identify Best Model
# -------------------------------------------------

best_model_name = comparison_df.iloc[0]['Model']
best_val_accuracy = comparison_df.iloc[0]['Val Accuracy']

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")








print("=" * 60)
print("DETAILED EVALUATION: LOGISTIC REGRESSION")
print("=" * 60)

# -------------------------------------------------
# 7.1: Get the Best Model and Predictions
# -------------------------------------------------

best_model = results['Logistic Regression']['model']
y_val_pred = results['Logistic Regression']['y_val_pred']

print(f"\nBest Model: Logistic Regression")
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f} ({accuracy_score(y_val, y_val_pred)*100:.2f}%)")

# -------------------------------------------------
# 7.2: Confusion Matrix
# -------------------------------------------------

# Calculate confusion matrix
cm = confusion_matrix(y_val, y_val_pred)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, 
            annot=True,           # Show numbers in cells
            fmt='d',              # Format as integers
            cmap='Blues',         # Blue color scheme
            xticklabels=['Low (0)', 'Medium (1)', 'High (2)', 'Very High (3)'],
            yticklabels=['Low (0)', 'Medium (1)', 'High (2)', 'Very High (3)'])

plt.title('Confusion Matrix - Logistic Regression', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Price Range', fontsize=12)
plt.ylabel('Actual Price Range', fontsize=12)
plt.tight_layout()
plt.show()

# Print confusion matrix as text for clarity
print("\n" + "=" * 60)
print("CONFUSION MATRIX (Text Format)")
print("=" * 60)
print("\nRows = Actual Price Range, Columns = Predicted Price Range\n")
print("           Pred 0  Pred 1  Pred 2  Pred 3")
for i, row in enumerate(cm):
    print(f"Actual {i}:   {row[0]:6d}  {row[1]:6d}  {row[2]:6d}  {row[3]:6d}")

# -------------------------------------------------
# 7.3: Classification Report
# -------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_val, y_val_pred, 
                            target_names=['Low (0)', 'Medium (1)', 'High (2)', 'Very High (3)']))

# -------------------------------------------------
# 7.4: Analyze Misclassifications
# -------------------------------------------------

print("=" * 60)
print("MISCLASSIFICATION ANALYSIS")
print("=" * 60)

# Find indices of misclassified samples
misclassified_indices = np.where(y_val_pred != y_val)[0]

print(f"\nTotal validation samples: {len(y_val)}")
print(f"Correctly classified:     {len(y_val) - len(misclassified_indices)}")
print(f"Misclassified:            {len(misclassified_indices)}")
print(f"Misclassification rate:   {len(misclassified_indices)/len(y_val)*100:.2f}%")

# Show examples of misclassifications
if len(misclassified_indices) > 0:
    print("\nExamples of Misclassified Samples:")
    print("-" * 50)
    
    # Get the original (unscaled) validation data for interpretation
    X_val_original = X_val.iloc[misclassified_indices[:5]]
    
    for i, idx in enumerate(misclassified_indices[:5]):
        actual = y_val.iloc[idx]
        predicted = y_val_pred[idx]
        print(f"\nSample {i+1}:")
        print(f"  Actual:    {actual}")
        print(f"  Predicted: {predicted}")
        print(f"  RAM:       {X_val_original.iloc[i]['ram']:.0f} MB")
        print(f"  Battery:   {X_val_original.iloc[i]['battery_power']:.0f} mAh")
        print(f"  Clock:     {X_val_original.iloc[i]['clock_speed']:.1f} GHz")
        print(f"  Int Mem:   {X_val_original.iloc[i]['int_memory']:.0f} GB")

# -------------------------------------------------
# 7.5: Feature Importance (Coefficients)
# -------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE - LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 60)

# Get feature names
feature_names = X.columns.tolist()

# Get coefficients for each class
coefficients = best_model.coef_

# Average the absolute coefficients across classes for overall importance
avg_importance = np.abs(coefficients).mean(axis=0)

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Avg Coefficient Magnitude': avg_importance
}).sort_values('Avg Coefficient Magnitude', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance_df.head(15)

plt.barh(range(len(top_features)), top_features['Avg Coefficient Magnitude'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Average Coefficient Magnitude', fontsize=12)
plt.title('Top 15 Most Important Features (Logistic Regression)', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()  # Highest importance at top
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 7.6: Per-Class Coefficient Analysis
# -------------------------------------------------

print("\n" + "=" * 60)
print("PER-CLASS COEFFICIENT ANALYSIS")
print("=" * 60)

# Create a DataFrame showing coefficients for each class
coef_df = pd.DataFrame(coefficients.T, 
                       columns=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
                       index=feature_names)

# Show RAM coefficients (most important feature)
print("\nCoefficients for RAM (most important feature):")
for i in range(4):
    print(f"  Class {i}: {coef_df.loc['ram', f'Class {i}']:.4f}")

print("\nInterpretation:")
print("  • Positive coefficient: Higher feature value → Higher probability of this class")
print("  • Negative coefficient: Higher feature value → Lower probability of this class")
print("  • Magnitude: Strength of the relationship")





# ==============================================
# STEP 8: PREDICTIONS ON TEST DATA
# ==============================================

print("=" * 60)
print("TEST DATA PREDICTIONS")
print("=" * 60)

# -------------------------------------------------
# 8.1: Make Predictions on Test Data
# -------------------------------------------------

# Use the best model to predict on test data
y_test_pred = best_model.predict(X_test_scaled)

print(f"\nPredictions made on {len(y_test_pred)} test samples.")

# Show distribution of predictions
unique, counts = np.unique(y_test_pred, return_counts=True)
print("\nPrediction Distribution on Test Data:")
for price_range, count in zip(unique, counts):
    percentage = (count / len(y_test_pred)) * 100
    print(f"  Price Range {price_range}: {count} phones ({percentage:.1f}%)")

# -------------------------------------------------
# 8.2: Create Submission DataFrame
# -------------------------------------------------

# Create a DataFrame with ID and predicted price range
submission = pd.DataFrame({
    'id': test_ids,
    'price_range': y_test_pred
})

print("\n" + "=" * 60)
print("SUBMISSION FILE PREVIEW")
print("=" * 60)
print("\nFirst 10 rows of submission file:")
print(submission.head(10))

print("\nLast 10 rows of submission file:")
print(submission.tail(10))

# -------------------------------------------------
# 8.3: Save Submission File
# -------------------------------------------------

# Save to CSV file (without index)
submission_file_path = 'data/submission.csv'
submission.to_csv(submission_file_path, index=False)

print("\n" + "=" * 60)
print("FILE SAVED")
print("=" * 60)
print(f"\n✅ Submission file saved to: '{submission_file_path}'")
print(f"   Total predictions: {len(submission)}")

# -------------------------------------------------
# 8.4: Save the Trained Model (Optional)
# -------------------------------------------------

import joblib

model_file_path = 'models/logistic_regression_model.pkl'

# Note: You need to create the 'models' folder first, or this may error
# To create folder programmatically:
import os
os.makedirs('models', exist_ok=True)

joblib.dump(best_model, model_file_path)
print(f"\n✅ Trained model saved to: '{model_file_path}'")

# Also save the scaler for future use
scaler_file_path = 'models/standard_scaler.pkl'
joblib.dump(scaler, scaler_file_path)
print(f"✅ StandardScaler saved to: '{scaler_file_path}'")

# -------------------------------------------------
# 8.5: Final Summary
# -------------------------------------------------

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(f"""
📊 DATA SUMMARY:
   • Training samples: 2000
   • Test samples:     1000
   • Features:         20
   • Target classes:   4 (0=Low, 1=Medium, 2=High, 3=Very High)

🏆 BEST MODEL:
   • Algorithm:        Logistic Regression
   • Validation Acc:   96.50%
   • Training Time:    0.014 seconds

📁 OUTPUT FILES:
   • Submission:       {submission_file_path}
   • Saved Model:      {model_file_path}
   • Saved Scaler:     {scaler_file_path}

🎯 PREDICTION DISTRIBUTION ON TEST SET:
   • Class 0: {counts[0]} phones ({counts[0]/len(y_test_pred)*100:.1f}%)
   • Class 1: {counts[1]} phones ({counts[1]/len(y_test_pred)*100:.1f}%)
   • Class 2: {counts[2]} phones ({counts[2]/len(y_test_pred)*100:.1f}%)
   • Class 3: {counts[3]} phones ({counts[3]/len(y_test_pred)*100:.1f}%)

✨ Project Complete! ✨
""")

# -------------------------------------------------
# 8.6: Visualize Test Predictions
# -------------------------------------------------

plt.figure(figsize=(10, 6))
sns.countplot(x=y_test_pred, palette='viridis')
plt.title('Predicted Price Range Distribution - Test Data', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Price Range', fontsize=12)
plt.ylabel('Number of Phones', fontsize=12)

# Add percentage labels
for i in range(4):
    count = np.sum(y_test_pred == i)
    percentage = count / len(y_test_pred) * 100
    plt.text(i, count + 5, f'{count}\n({percentage:.1f}%)', 
             ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()







