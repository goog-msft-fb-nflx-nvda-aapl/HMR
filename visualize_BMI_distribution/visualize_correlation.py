import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# def analyze_correlations(csv_path):
#     # Read the CSV file
#     df = pd.DataFrame(pd.read_csv(csv_path))
    
#     # Define error metrics to analyze
#     error_metrics = ['p2p_error', 'height_error', 'chest_error', 
#                     'waist_error', 'hips_error', 'mass_error']
    
#     # Create correlation matrix
#     corr_data = df[error_metrics + ['gt_bmi']].corr()
    
#     # Set up the matplotlib figure for correlation heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, fmt='.2f')
#     plt.title('Correlation Matrix: Error Metrics vs BMI')
#     plt.tight_layout()
#     plt.savefig('correlation_heatmap.png')
#     plt.close()
    
#     # Create scatter plots with regression lines
#     fig, axes = plt.subplots(2, 3, figsize=(20, 12))
#     axes = axes.ravel()
    
#     for idx, metric in enumerate(error_metrics):
#         # Calculate correlation coefficient and p-value
#         correlation, p_value = stats.pearsonr(df[metric], df['gt_bmi'])
        
#         # Create scatter plot
#         sns.scatterplot(data=df, x='gt_bmi', y=metric, ax=axes[idx], alpha=0.5)
#         sns.regplot(data=df, x='gt_bmi', y=metric, ax=axes[idx], 
#                    scatter=False, color='red')
        
#         # Add correlation information
#         axes[idx].set_title(f'{metric} vs BMI\nr={correlation:.3f}, p={p_value:.3e}')
#         axes[idx].set_xlabel('Ground Truth BMI')
#         axes[idx].set_ylabel(f'{metric}')
        
#         # Add trend description
#         trend_text = (
#             'Positive Correlation' if correlation > 0 
#             else 'Negative Correlation' if correlation < 0 
#             else 'No Correlation'
#         )
#         strength_text = (
#             'Strong' if abs(correlation) > 0.5 
#             else 'Moderate' if abs(correlation) > 0.3 
#             else 'Weak'
#         )
#         axes[idx].text(0.05, 0.95, f'{strength_text} {trend_text}',
#                       transform=axes[idx].transAxes,
#                       verticalalignment='top')
    
#     plt.tight_layout()
#     plt.savefig('error_vs_bmi_scatter.png')
#     plt.close()
    
#     # Calculate and print summary statistics
#     print("\nSummary Statistics:")
#     print("-" * 50)
#     for metric in error_metrics:
#         correlation, p_value = stats.pearsonr(df[metric], df['gt_bmi'])
#         print(f"\n{metric} vs BMI:")
#         print(f"Correlation coefficient: {correlation:.3f}")
#         print(f"P-value: {p_value:.3e}")
#         print(f"Mean {metric}: {df[metric].mean():.2f}")
#         print(f"Std {metric}: {df[metric].std():.2f}")
        
#         # Calculate error statistics by BMI category
#         df['bmi_category'] = pd.cut(df['gt_bmi'], 
#                                   bins=[0, 18.5, 25, 30, float('inf')],
#                                   labels=['Underweight', 'Normal', 
#                                         'Overweight', 'Obese'])
        
#         print("\nError statistics by BMI category:")
#         print(df.groupby('bmi_category')[metric].describe()[['mean', 'std']])

#     # Create box plots for errors across BMI categories
#     plt.figure(figsize=(15, 8))
#     for idx, metric in enumerate(error_metrics):
#         plt.subplot(2, 3, idx+1)
#         sns.boxplot(data=df, x='bmi_category', y=metric)
#         plt.xticks(rotation=45)
#         plt.title(f'{metric} by BMI Category')
    
#     plt.tight_layout()
#     plt.savefig('/home/iismtl519-2/Desktop/error_by_bmi_category.png')
#     plt.close()

# if __name__ == "__main__":
#     # Replace with your CSV file path
#     csv_path = "/home/iismtl519-2/Desktop/refine/hbw/betas/PR_orig1_aug0/smpl/am/smplMeshes_detailed_metrics.csv"
#     analyze_correlations(csv_path)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def analyze_errors_vs_bmi(csv_path):
    # Read the CSV file
    df = pd.DataFrame(pd.read_csv(csv_path))
    
    # Calculate averaged measurement error
    measurement_errors = ['height_error', 'chest_error', 'waist_error', 'hips_error']
    df['avg_measurement_error'] = df[measurement_errors].mean(axis=1)
    
    # Define error metrics to analyze
    error_metrics = ['p2p_error', 'avg_measurement_error']
    
    # Create BMI categories
    df['bmi_category'] = pd.cut(df['gt_bmi'], 
                               bins=[0, 18.5, 25, 30, float('inf')],
                               labels=['Underweight', 'Normal', 
                                     'Overweight', 'Obese'])
    
    # Remove categories with no data points
    category_counts = df['bmi_category'].value_counts()
    populated_categories = category_counts[category_counts > 0].index
    df_filtered = df[df['bmi_category'].isin(populated_categories)]
    
    # Create visualizations for each error metric
    for metric in error_metrics:
        metric_name = 'Point-to-Point Error' if metric == 'p2p_error' else 'Average Measurement Error'
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{metric_name} vs BMI Analysis', fontsize=16, y=0.95)
        
        # 1. Scatter plot with regression line
        correlation, p_value = stats.pearsonr(df[metric], df['gt_bmi'])
        sns.regplot(data=df, x='gt_bmi', y=metric, scatter=True,
                   scatter_kws={'alpha':0.5}, line_kws={'color': 'red'}, ax=ax1)
        ax1.set_title(f'Scatter Plot with Regression Line\nr={correlation:.3f}, p={p_value:.3e}')
        ax1.set_xlabel('Ground Truth BMI')
        ax1.set_ylabel(f'{metric_name} (mm)')
        
        # 2. Box plot by BMI category
        sns.boxplot(data=df_filtered, x='bmi_category', y=metric, ax=ax2)
        ax2.set_title('Box Plot by BMI Category')
        ax2.set_xlabel('BMI Category')
        ax2.set_ylabel(f'{metric_name} (mm)')
        
        # 3. Violin plot by BMI category
        sns.violinplot(data=df_filtered, x='bmi_category', y=metric, ax=ax3)
        ax3.set_title('Violin Plot by BMI Category')
        ax3.set_xlabel('BMI Category')
        ax3.set_ylabel(f'{metric_name} (mm)')
        
        # 4. Summary statistics table
        ax4.axis('off')
        stats_text = "Summary Statistics:\n\n"
        
        # Overall statistics
        stats_text += f"Overall Statistics:\n"
        stats_text += f"Mean: {df[metric].mean():.2f} mm\n"
        stats_text += f"Std: {df[metric].std():.2f} mm\n"
        stats_text += f"Correlation with BMI: {correlation:.3f}\n"
        stats_text += f"P-value: {p_value:.3e}\n\n"
        
        if metric == 'avg_measurement_error':
            stats_text += "Component Errors (mean ± std):\n"
            for err in measurement_errors:
                stats_text += f"{err.replace('_error', '')}: "
                stats_text += f"{df[err].mean():.2f} ± {df[err].std():.2f} mm\n"
            stats_text += "\n"
        
        # Statistics by BMI category
        stats_text += "Statistics by BMI Category:\n"
        category_stats = df_filtered.groupby('bmi_category')[metric].agg(['mean', 'std', 'count'])
        
        for category in populated_categories:
            stats_text += f"\n{category}:\n"
            stats_text += f"  Count: {category_stats.loc[category, 'count']}\n"
            stats_text += f"  Mean: {category_stats.loc[category, 'mean']:.2f} mm\n"
            stats_text += f"  Std: {category_stats.loc[category, 'std']:.2f} mm\n"
        
        ax4.text(0, 1, stats_text, va='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f'/home/iismtl519-2/Desktop/{metric}_bmi_analysis.png')
        plt.close()

        # Print statistics to console
        print(f"\n{'='*50}")
        print(f"Statistics for {metric_name}:")
        print(f"{'='*50}")
        print("\nCorrelation with BMI:", correlation)
        print("P-value:", p_value)
        print("\nStatistics by BMI category:")
        print(category_stats)

if __name__ == "__main__":
    # Replace with your CSV file path
    csv_path = "/home/iismtl519-2/Desktop/refine/hbw/betas/PR_orig1_aug0/smpl/am/smplMeshes_detailed_metrics.csv"
    analyze_errors_vs_bmi(csv_path)