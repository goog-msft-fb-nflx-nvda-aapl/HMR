import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def create_sports_visualizations(df, output_dir='sports_visualizations'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set the style for all plots - using a default matplotlib style
    plt.style.use('default')
    
    # Set a consistent figure size and font size
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['font.size'] = 10
    
    # 1. Distribution of Sports by Gender
    plt.figure()
    sns.countplot(data=df, x='Sport Type', hue='genders')
    plt.xticks(rotation=45)
    plt.title('Distribution of Athletes by Sport and Gender')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sports_gender_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. BMI Distribution by Sport Type
    plt.figure()
    sns.boxplot(data=df, x='Sport Type', y='BMI ( kg / (m^2) )')
    plt.xticks(rotation=45)
    plt.title('BMI Distribution by Sport Type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bmi_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Height vs Mass Scatter Plot
    plt.figure()
    sns.scatterplot(data=df, x='Height (m)', y='Mass (kg)', 
                    hue='Sport Type', style='genders')
    plt.title('Height vs Mass by Sport Type and Gender')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/height_mass_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation Heatmap for Body Measurements
    measurements = ['Height (m)', 'Chest Circumference (m)', 
                   'Waist Circumference (m)', 'Hip Circumference (m)', 
                   'Mass (kg)', 'BMI ( kg / (m^2) )']
    plt.figure(figsize=(10, 8))  # Slightly different size for the heatmap
    sns.heatmap(df[measurements].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Body Measurements')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Body Proportions by Sport Type
    measurements_to_plot = ['Chest Circumference (m)', 
                          'Waist Circumference (m)', 
                          'Hip Circumference (m)']
    
    plt.figure()
    df_melted = pd.melt(df, id_vars=['Sport Type'], 
                        value_vars=measurements_to_plot)
    sns.boxplot(data=df_melted, x='Sport Type', y='value', hue='variable')
    plt.xticks(rotation=45)
    plt.title('Body Proportions by Sport Type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/body_proportions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1. Distribution of Sports by Gender
    plt.figure()
    sns.countplot(data=df, x='Sport Type', hue='genders')
    plt.xticks(rotation=45)
    plt.title('Distribution of Athletes by Sport and Gender')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sports_gender_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # New gender-focused visualizations
    
    # 2. BMI Distribution by Gender
    plt.figure()
    sns.violinplot(data=df, x='genders', y='BMI ( kg / (m^2) )')
    plt.title('BMI Distribution by Gender')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gender_bmi_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Height Distribution by Gender
    plt.figure()
    sns.boxplot(data=df, x='genders', y='Height (m)')
    plt.title('Height Distribution by Gender')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gender_height_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Body Measurements by Gender
    measurements = ['Chest Circumference (m)', 'Waist Circumference (m)', 'Hip Circumference (m)']
    plt.figure()
    df_melted = pd.melt(df, id_vars=['genders'], value_vars=measurements)
    sns.boxplot(data=df_melted, x='variable', y='value', hue='genders')
    plt.title('Body Measurements by Gender')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gender_body_measurements.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Mass Distribution by Gender for each Sport
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x='Sport Type', y='Mass (kg)', hue='genders')
    plt.title('Mass Distribution by Sport and Gender')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gender_mass_by_sport.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Gender-specific correlation heatmaps
    for gender in df['genders'].unique():
        measurements = ['Height (m)', 'Chest Circumference (m)', 
                       'Waist Circumference (m)', 'Hip Circumference (m)', 
                       'Mass (kg)', 'BMI ( kg / (m^2) )']
        
        plt.figure(figsize=(10, 8))
        gender_df = df[df['genders'] == gender]
        sns.heatmap(gender_df[measurements].corr(), 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0)
        plt.title(f'Correlation Heatmap of Body Measurements - {gender}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap_{gender}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Height to Mass Ratio by Gender
    plt.figure()
    df['height_mass_ratio'] = df['Height (m)'] / df['Mass (kg)']
    sns.boxplot(data=df, x='Sport Type', y='height_mass_ratio', hue='genders')
    plt.title('Height to Mass Ratio by Sport and Gender')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gender_height_mass_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All visualizations have been saved to the '{output_dir}' directory.")


df = pd.read_csv('ssp3d.csv')
create_sports_visualizations(df)