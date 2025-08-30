import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class WineQualityAnalyzer:
    """
    Comprehensive Wine Quality Analysis System
    Handles both classification and regression tasks with feature selection and outlier detection
    """
    
    def __init__(self, data_path=None):
        self.data = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol', 'quality'
        ]
        
    def load_sample_data(self):
        """Generate sample wine quality data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic wine data based on typical ranges
        data = {
            'fixed_acidity': np.random.normal(8.3, 1.7, n_samples),
            'volatile_acidity': np.random.gamma(2, 0.15, n_samples),
            'citric_acid': np.random.exponential(0.3, n_samples),
            'residual_sugar': np.random.exponential(2.5, n_samples),
            'chlorides': np.random.gamma(2, 0.04, n_samples),
            'free_sulfur_dioxide': np.random.normal(15, 10, n_samples),
            'total_sulfur_dioxide': np.random.normal(46, 32, n_samples),
            'density': np.random.normal(0.996, 0.002, n_samples),
            'pH': np.random.normal(3.3, 0.15, n_samples),
            'sulphates': np.random.normal(0.66, 0.17, n_samples),
            'alcohol': np.random.normal(10.4, 1.1, n_samples)
        }
        
        # Create quality scores based on feature relationships
        quality_score = (
            (data['alcohol'] - 8) * 0.5 +
            (12 - data['volatile_acidity'] * 20) * 0.3 +
            (data['citric_acid']) * 2 +
            (data['sulphates'] - 0.4) * 3 +
            np.random.normal(0, 1, n_samples)
        )
        
        # Convert to 0-10 scale and round
        data['quality'] = np.clip(np.round(quality_score + 6), 0, 10).astype(int)
        
        self.data = pd.DataFrame(data)
        print("Sample wine quality dataset generated successfully!")
        return self.data
    
    def load_data(self, file_path):
        """Load wine quality data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully! Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"File {file_path} not found. Generating sample data instead.")
            return self.load_sample_data()
    
    def explore_data(self):
        """Comprehensive data exploration and visualization"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("=" * 50)
        print("WINE QUALITY DATASET EXPLORATION")
        print("=" * 50)
        
        # Basic info
        print("\n1. DATASET OVERVIEW")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {list(self.data.columns[:-1])}")
        print(f"Target variable: {self.data.columns[-1]}")
        
        # Statistical summary
        print("\n2. STATISTICAL SUMMARY")
        print(self.data.describe())
        
        # Missing values
        print("\n3. MISSING VALUES")
        missing = self.data.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")
        
        # Quality distribution
        print("\n4. QUALITY DISTRIBUTION")
        quality_dist = self.data['quality'].value_counts().sort_index()
        print(quality_dist)
        
        # Create visualizations
        self._create_visualizations()
        
    def _create_visualizations(self):
        """Create comprehensive and readable visualizations with proper spacing"""
        # Set style for better appearance
        plt.style.use('default')
        plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})
        
        # Get numerical data only for correlation
        numerical_data = self.data.select_dtypes(include=[np.number])
        
        # Create Figure 1: Basic Analysis (2x2 layout)
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
        fig1.suptitle('Wine Quality Analysis - Basic Overview', fontsize=18, fontweight='bold', y=0.95)
        
        # 1. Quality distribution
        quality_counts = numerical_data['quality'].value_counts().sort_index()
        bars = axes1[0, 0].bar(quality_counts.index, quality_counts.values, 
                              color='lightcoral', edgecolor='darkred', linewidth=1.2, alpha=0.8)
        axes1[0, 0].set_title('Quality Score Distribution', fontsize=14, fontweight='bold')
        axes1[0, 0].set_xlabel('Quality Score', fontsize=12)
        axes1[0, 0].set_ylabel('Number of Wines', fontsize=12)
        axes1[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes1[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Quality vs Alcohol scatter plot
        scatter1 = axes1[0, 1].scatter(numerical_data['alcohol'], numerical_data['quality'], 
                                      c=numerical_data['volatile_acidity'], cmap='viridis', 
                                      alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        axes1[0, 1].set_title('Quality vs Alcohol Content\n(colored by Volatile Acidity)', 
                             fontsize=14, fontweight='bold')
        axes1[0, 1].set_xlabel('Alcohol Content (%)', fontsize=12)
        axes1[0, 1].set_ylabel('Quality Score', fontsize=12)
        axes1[0, 1].grid(True, alpha=0.3)
        cbar1 = plt.colorbar(scatter1, ax=axes1[0, 1], shrink=0.8)
        cbar1.set_label('Volatile Acidity', fontsize=11)
        
        # 3. Feature correlation with quality
        feature_corr = numerical_data.corr()['quality'].abs().sort_values(ascending=False)[1:]
        colors_grad = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(feature_corr)))
        bars2 = axes1[1, 0].barh(range(len(feature_corr)), feature_corr.values, color=colors_grad)
        
        axes1[1, 0].set_yticks(range(len(feature_corr)))
        axes1[1, 0].set_yticklabels([name.replace('_', ' ').title() for name in feature_corr.index], 
                                   fontsize=10)
        axes1[1, 0].set_title('Feature Correlation with Quality', fontsize=14, fontweight='bold')
        axes1[1, 0].set_xlabel('Absolute Correlation', fontsize=12)
        axes1[1, 0].grid(axis='x', alpha=0.3)
        
        # Add correlation values
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            axes1[1, 0].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 4. Quality categories pie chart
        quality_categories = pd.cut(numerical_data['quality'], 
                                   bins=[0, 4, 6, 8, 10], 
                                   labels=['Poor (0-4)', 'Below Avg (5-6)', 
                                          'Good (7-8)', 'Excellent (9-10)'])
        category_counts = quality_categories.value_counts()
        
        colors_pie = ['#ff7f7f', '#ffb366', '#90ee90', '#ffd700']
        wedges, texts, autotexts = axes1[1, 1].pie(category_counts.values, 
                                                  labels=category_counts.index, 
                                                  autopct='%1.1f%%', 
                                                  startangle=90,
                                                  colors=colors_pie,
                                                  explode=[0.05, 0.05, 0.05, 0.05])
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        
        axes1[1, 1].set_title('Wine Quality Categories', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Create Figure 2: Correlation Matrix (standalone)
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        corr_matrix = numerical_data.corr()
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax2,
                   linewidths=0.5)
        ax2.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        ax2.tick_params(axis='both', which='major', labelsize=11)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Create Figure 3: Detailed Feature Analysis
        fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
        fig3.suptitle('Detailed Feature Analysis', fontsize=18, fontweight='bold', y=0.95)
        
        # 1. Box plots for alcohol by quality
        quality_levels = sorted(numerical_data['quality'].unique())
        alcohol_by_quality = [numerical_data[numerical_data['quality'] == q]['alcohol'].values 
                             for q in quality_levels]
        
        bp1 = axes3[0, 0].boxplot(alcohol_by_quality, labels=quality_levels,
                                 patch_artist=True, notch=True)
        
        # Color the boxes with gradient
        box_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bp1['boxes'])))
        for patch, color in zip(bp1['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes3[0, 0].set_title('Alcohol Content by Quality Score', fontsize=14, fontweight='bold')
        axes3[0, 0].set_xlabel('Quality Score', fontsize=12)
        axes3[0, 0].set_ylabel('Alcohol Content (%)', fontsize=12)
        axes3[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. pH vs Fixed Acidity
        scatter3 = axes3[0, 1].scatter(numerical_data['fixed_acidity'], numerical_data['pH'], 
                                      c=numerical_data['quality'], cmap='RdYlGn', 
                                      alpha=0.7, s=30, edgecolors='black', linewidth=0.3)
        axes3[0, 1].set_title('pH vs Fixed Acidity\n(Quality Color-Coded)', fontsize=14, fontweight='bold')
        axes3[0, 1].set_xlabel('Fixed Acidity (g/L)', fontsize=12)
        axes3[0, 1].set_ylabel('pH Level', fontsize=12)
        axes3[0, 1].grid(True, alpha=0.3)
        cbar3_1 = plt.colorbar(scatter3, ax=axes3[0, 1], shrink=0.8)
        cbar3_1.set_label('Quality Score', fontsize=11)
        
        # 3. Density vs Alcohol with trend line
        axes3[1, 0].scatter(numerical_data['density'], numerical_data['alcohol'], 
                           alpha=0.7, color='steelblue', s=30, edgecolors='navy', linewidth=0.3)
        
        # Add trend line
        z = np.polyfit(numerical_data['density'], numerical_data['alcohol'], 1)
        p = np.poly1d(z)
        trend_line = axes3[1, 0].plot(numerical_data['density'], p(numerical_data['density']), 
                                     "r--", alpha=0.8, linewidth=3, label=f'Trend: R²={np.corrcoef(numerical_data["density"], numerical_data["alcohol"])[0,1]**2:.3f}')
        
        axes3[1, 0].set_title('Alcohol vs Density Relationship', fontsize=14, fontweight='bold')
        axes3[1, 0].set_xlabel('Density (g/cm³)', fontsize=12)
        axes3[1, 0].set_ylabel('Alcohol Content (%)', fontsize=12)
        axes3[1, 0].grid(True, alpha=0.3)
        axes3[1, 0].legend()
        
        # 4. Sulfur dioxide relationship
        scatter4 = axes3[1, 1].scatter(numerical_data['free_sulfur_dioxide'], 
                                      numerical_data['total_sulfur_dioxide'],
                                      c=numerical_data['quality'], cmap='plasma', 
                                      alpha=0.7, s=30, edgecolors='black', linewidth=0.3)
        axes3[1, 1].set_title('Free vs Total SO₂\n(Quality Color-Coded)', fontsize=14, fontweight='bold')
        axes3[1, 1].set_xlabel('Free SO₂ (mg/L)', fontsize=12)
        axes3[1, 1].set_ylabel('Total SO₂ (mg/L)', fontsize=12)
        axes3[1, 1].grid(True, alpha=0.3)
        cbar4 = plt.colorbar(scatter4, ax=axes3[1, 1], shrink=0.8)
        cbar4.set_label('Quality Score', fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
        # Create Figure 4: Feature Distributions
        self._create_feature_distributions(numerical_data)
    
    def _create_feature_distributions(self, numerical_data):
        """Create detailed feature distribution plots"""
        # Select top 6 most correlated features with quality
        top_features = numerical_data.corr()['quality'].abs().sort_values(ascending=False)[1:7]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribution of Key Features by Quality Score', fontsize=18, fontweight='bold', y=0.95)
        
        quality_levels = sorted(numerical_data['quality'].unique())
        
        for i, feature in enumerate(top_features.index):
            row, col = i // 3, i % 3
            
            # Create violin plot data
            violin_data = [numerical_data[numerical_data['quality'] == q][feature].dropna().values 
                          for q in quality_levels]
            
            # Only plot if we have data
            valid_positions = []
            valid_data = []
            valid_labels = []
            
            for j, (pos, data) in enumerate(zip(quality_levels, violin_data)):
                if len(data) > 0:
                    valid_positions.append(pos)
                    valid_data.append(data)
                    valid_labels.append(str(pos))
            
            if valid_data:
                parts = axes[row, col].violinplot(valid_data, positions=valid_positions,
                                                 showmeans=True, showmedians=True, widths=0.7)
                
                # Color the violins
                for pc in parts['bodies']:
                    pc.set_facecolor('lightblue')
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('navy')
                    pc.set_linewidth(1.5)
            
            axes[row, col].set_title(f'{feature.replace("_", " ").title()}', 
                                   fontsize=14, fontweight='bold')
            axes[row, col].set_xlabel('Quality Score', fontsize=12)
            axes[row, col].set_ylabel(f'{feature.replace("_", " ").title()}', fontsize=12)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xticks(quality_levels)
            
            # Add correlation info
            corr_value = numerical_data[feature].corr(numerical_data['quality'])
            axes[row, col].text(0.02, 0.98, f'r = {corr_value:.3f}', 
                               transform=axes[row, col].transAxes,
                               fontsize=11, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def detect_outliers(self):
        """Detect outliers using Isolation Forest"""
        print("\n" + "=" * 50)
        print("OUTLIER DETECTION")
        print("=" * 50)
        
        features = self.data.drop('quality', axis=1)
        
        # Isolation Forest for outlier detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(features)
        
        n_outliers = np.sum(outliers == -1)
        print(f"Number of outliers detected: {n_outliers} ({n_outliers/len(self.data)*100:.1f}%)")
        
        # Show outlier quality distribution
        outlier_mask = outliers == -1
        print(f"Quality distribution of outliers:")
        print(self.data[outlier_mask]['quality'].value_counts().sort_index())
        
        return outlier_mask
    
    def feature_selection(self, k=8):
        """Perform feature selection using SelectKBest"""
        print("\n" + "=" * 50)
        print("FEATURE SELECTION")
        print("=" * 50)
        
        X = self.data.drop('quality', axis=1)
        y = self.data['quality']
        
        # For classification
        selector_clf = SelectKBest(score_func=f_classif, k=k)
        X_selected_clf = selector_clf.fit_transform(X, y)
        selected_features_clf = X.columns[selector_clf.get_support()].tolist()
        
        # For regression
        selector_reg = SelectKBest(score_func=f_regression, k=k)
        X_selected_reg = selector_reg.fit_transform(X, y)
        selected_features_reg = X.columns[selector_reg.get_support()].tolist()
        
        print(f"Top {k} features for classification: {selected_features_clf}")
        print(f"Top {k} features for regression: {selected_features_reg}")
        
        # Feature scores
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Classification_Score': selector_clf.scores_,
            'Regression_Score': selector_reg.scores_
        }).sort_values('Classification_Score', ascending=False)
        
        print("\nFeature Scores:")
        print(feature_scores)
        
        return selected_features_clf, selected_features_reg
    
    def classification_analysis(self):
        """Perform classification analysis"""
        print("\n" + "=" * 50)
        print("CLASSIFICATION ANALYSIS")
        print("=" * 50)
        
        # Prepare data for classification
        X = self.data.drop('quality', axis=1)
        y = self.data['quality']
        
        # Create quality categories for better classification
        y_categorical = pd.cut(y, bins=[0, 4, 6, 10], labels=['Poor', 'Average', 'Excellent'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Models to compare
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{name} Results:")
            
            # Train model
            if name == 'Random Forest':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Cross-validation score
            cv_data = X_train_scaled if name != 'Random Forest' else X_train
            cv_scores = cross_val_score(model, cv_data, y_train, cv=5)
            
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            results[name] = {
                'model': model,
                'cv_score': cv_scores.mean(),
                'predictions': y_pred
            }
        
        # Feature importance for Random Forest
        rf_model = results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nRandom Forest Feature Importance:")
        print(feature_importance)
        
        # Create feature importance visualization
        self._plot_feature_importance(feature_importance)
        
        return results
    
    def _plot_regression_results(self, results, y_test):
        """Create detailed regression results visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Regression Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(results.keys())
        colors = ['coral', 'lightblue', 'lightgreen']
        
        # 1. Model performance comparison
        rmse_scores = [results[model]['rmse'] for model in models]
        r2_scores = [results[model]['r2'] for model in models]
        
        x_pos = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, rmse_scores, width, label='RMSE', 
                      color='lightcoral', edgecolor='darkred')
        ax_twin = axes[0, 0].twinx()
        ax_twin.bar(x_pos + width/2, r2_scores, width, label='R² Score', 
                   color='lightblue', edgecolor='darkblue')
        
        axes[0, 0].set_xlabel('Models', fontsize=12)
        axes[0, 0].set_ylabel('RMSE', fontsize=12)
        ax_twin.set_ylabel('R² Score', fontsize=12)
        axes[0, 0].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Actual vs Predicted for best model
        best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
        y_pred_best = results[best_model]['predictions']
        
        axes[0, 1].scatter(y_test, y_pred_best, alpha=0.7, color='green', s=40,
                          edgecolors='darkgreen', linewidth=0.5)
        
        # Perfect prediction line
        min_val, max_val = min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        axes[0, 1].set_xlabel('Actual Quality', fontsize=12)
        axes[0, 1].set_ylabel('Predicted Quality', fontsize=12)
        axes[0, 1].set_title(f'Actual vs Predicted ({best_model})', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add R² score to the plot
        r2_best = results[best_model]['r2']
        axes[0, 1].text(0.05, 0.95, f'R² = {r2_best:.3f}', transform=axes[0, 1].transAxes,
                       fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Residuals plot
        residuals = y_test - y_pred_best
        axes[1, 0].scatter(y_pred_best, residuals, alpha=0.7, color='orange', s=40,
                          edgecolors='darkorange', linewidth=0.5)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Predicted Quality', fontsize=12)
        axes[1, 0].set_ylabel('Residuals', fontsize=12)
        axes[1, 0].set_title(f'Residuals Plot ({best_model})', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error distribution
        axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='mediumpurple', 
                       edgecolor='darkviolet', density=True)
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residuals', fontsize=12)
        axes[1, 1].set_ylabel('Density', fontsize=12)
        axes[1, 1].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(residuals)
        std_error = np.std(residuals)
        axes[1, 1].text(0.05, 0.95, f'Mean: {mean_error:.3f}\nStd: {std_error:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def _plot_feature_importance(self, feature_importance):
        """Create a detailed feature importance plot"""
        plt.figure(figsize=(12, 8))
        
        # Sort by importance
        sorted_features = feature_importance.sort_values('Importance', ascending=True)
        
        # Create horizontal bar plot
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_features)))
        bars = plt.barh(range(len(sorted_features)), sorted_features['Importance'], 
                       color=colors, edgecolor='black', linewidth=0.8)
        
        # Customize the plot
        plt.yticks(range(len(sorted_features)), 
                  [feat.replace('_', ' ').title() for feat in sorted_features['Feature']], 
                  fontsize=12)
        plt.xlabel('Feature Importance', fontsize=14, fontweight='bold')
        plt.title('Random Forest Feature Importance for Wine Quality Prediction', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def regression_analysis(self):
        """Perform regression analysis"""
        print("\n" + "=" * 50)
        print("REGRESSION ANALYSIS")
        print("=" * 50)
        
        # Prepare data for regression
        X = self.data.drop('quality', axis=1)
        y = self.data['quality']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Models to compare
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(kernel='rbf')
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{name} Results:")
            
            # Train model
            if name == 'Random Forest':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"RMSE: {rmse:.3f}")
            print(f"R² Score: {r2:.3f}")
            print(f"Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.3f}")
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
        
        # Create regression visualization
        self._plot_regression_results(results, y_test)
        
        return results
    
    def advanced_analysis(self):
        """Perform advanced analysis including PCA and hyperparameter tuning"""
        print("\n" + "=" * 50)
        print("ADVANCED ANALYSIS")
        print("=" * 50)
        
        X = self.data.drop('quality', axis=1)
        y = self.data['quality']
        
        # PCA Analysis
        print("\n1. PRINCIPAL COMPONENT ANALYSIS")
        pca = PCA()
        X_pca = pca.fit_transform(self.scaler.fit_transform(X))
        
        # Explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print(f"First 3 components explain {cumulative_var[2]:.1%} of variance")
        print("Component loadings (first 3 components):")
        
        components_df = pd.DataFrame(
            pca.components_[:3].T,
            columns=['PC1', 'PC2', 'PC3'],
            index=X.columns
        )
        print(components_df)
        
        # Hyperparameter tuning for Random Forest
        print("\n2. HYPERPARAMETER TUNING (Random Forest)")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42)
        y_categorical = pd.cut(y, bins=[0, 4, 6, 10], labels=['Poor', 'Average', 'Excellent'])
        
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y_categorical)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        return {
            'pca': pca,
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_
        }
    
    def create_quality_predictor(self, features_to_use=None):
        """Create a wine quality predictor"""
        print("\n" + "=" * 50)
        print("WINE QUALITY PREDICTOR")
        print("=" * 50)
        
        X = self.data.drop('quality', axis=1)
        y = self.data['quality']
        
        if features_to_use:
            X = X[features_to_use]
        
        # Train final model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        final_model = RandomForestRegressor(n_estimators=200, random_state=42)
        final_model.fit(X_train, y_train)
        
        # Model performance
        train_score = final_model.score(X_train, y_train)
        test_score = final_model.score(X_test, y_test)
        
        print(f"Training R² Score: {train_score:.3f}")
        print(f"Testing R² Score: {test_score:.3f}")
        
        # Example predictions
        print("\nExample Predictions:")
        sample_indices = np.random.choice(len(X_test), 5, replace=False)
        
        for i, idx in enumerate(sample_indices):
            actual = y_test.iloc[idx]
            predicted = final_model.predict(X_test.iloc[idx:idx+1])[0]
            print(f"Wine {i+1}: Actual={actual}, Predicted={predicted:.1f}")
        
        return final_model
    
    def run_complete_analysis(self, data_path=None):
        """Run the complete wine quality analysis pipeline"""
        print("🍷 WINE QUALITY ANALYSIS SYSTEM")
        print("Analyzing physicochemical properties and sensory quality ratings")
        
        # Load data
        if data_path:
            self.load_data(data_path)
        else:
            self.load_sample_data()
        
        # Exploratory data analysis
        self.explore_data()
        
        # Outlier detection
        outlier_mask = self.detect_outliers()
        
        # Feature selection
        clf_features, reg_features = self.feature_selection()
        
        # Classification analysis
        clf_results = self.classification_analysis()
        
        # Regression analysis
        reg_results = self.regression_analysis()
        
        # Advanced analysis
        advanced_results = self.advanced_analysis()
        
        # Final predictor
        final_model = self.create_quality_predictor()
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        print("Key findings:")
        print("- Dataset contains wine quality scores from 0-10")
        print("- Multiple physicochemical features influence quality")
        print("- Classification into Poor/Average/Excellent categories possible")
        print("- Regression models can predict continuous quality scores")
        print("- Feature selection helps identify most important variables")
        print("- Outlier detection can identify exceptional wines")
        
        return {
            'data': self.data,
            'outliers': outlier_mask,
            'classification_results': clf_results,
            'regression_results': reg_results,
            'advanced_results': advanced_results,
            'final_model': final_model
        }

# Example usage and demonstration
def main():
    """Main function to demonstrate the wine quality analysis"""
    
    # Initialize analyzer
    analyzer = WineQualityAnalyzer()
    
    # Run complete analysis
    # To use your own CSV file, pass the file path:
    # results = analyzer.run_complete_analysis('your_wine_data.csv')
    
    # For demonstration with sample data:
    results = analyzer.run_complete_analysis()
    
    # Interactive prediction example
    print("\n" + "=" * 50)
    print("INTERACTIVE PREDICTION EXAMPLE")
    print("=" * 50)
    
    # Example wine properties for prediction
    example_wine = {
        'fixed_acidity': 7.4,
        'volatile_acidity': 0.7,
        'citric_acid': 0.0,
        'residual_sugar': 1.9,
        'chlorides': 0.076,
        'free_sulfur_dioxide': 11.0,
        'total_sulfur_dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }
    
    # Create DataFrame for prediction
    wine_df = pd.DataFrame([example_wine])
    predicted_quality = results['final_model'].predict(wine_df)[0]
    
    print("Example wine properties:")
    for prop, value in example_wine.items():
        print(f"  {prop}: {value}")
    
    print(f"\nPredicted quality score: {predicted_quality:.1f}/10")
    
    if predicted_quality < 5:
        category = "Poor"
    elif predicted_quality < 7:
        category = "Average"
    else:
        category = "Excellent"
    
    print(f"Quality category: {category}")

# Run the analysis
if __name__ == "__main__":
    main()