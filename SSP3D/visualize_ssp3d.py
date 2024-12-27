import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import re

def create_safe_filename(text):
    """Create a safe filename by removing or replacing problematic characters"""
    # Replace special characters with underscores
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', text)
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name

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

    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set the style for all plots
    plt.style.use('default')
    
    # Set a consistent figure size and font size
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['font.size'] = 10
    
    # Define a color palette for genders
    color_palette = sns.color_palette("Set2")
    gender_colors = dict(zip(df['genders'].unique(), color_palette))
    
    # Add distribution plots for all measurements
    measurements = {
        'Height (m)': 'Height Distribution',
        'Mass (kg)': 'Mass Distribution',
        'BMI ( kg / (m^2) )': 'BMI Distribution',
        'Chest Circumference (m)': 'Chest Circumference Distribution',
        'Waist Circumference (m)': 'Waist Circumference Distribution',
        'Hip Circumference (m)': 'Hip Circumference Distribution'
    }
    
    # Create distribution plot for each measurement
    for col, title in measurements.items():
        plt.figure(figsize=(12, 6))
        
        # Create the main distribution plot
        sns.kdeplot(data=df, x=col, hue='genders', fill=True, alpha=0.5, palette=gender_colors)
        
        # Add individual points at the bottom (rug plot)
        sns.rugplot(data=df, x=col, hue='genders', alpha=0.5, palette=gender_colors)
        
        # Add mean lines for each gender
        for gender in df['genders'].unique():
            mean_val = df[df['genders'] == gender][col].mean()
            plt.axvline(x=mean_val, 
                       color=gender_colors[gender],
                       linestyle='--',
                       alpha=0.8,
                       label=f'{gender} Mean')
        
        plt.title(f'{title} by Gender')
        plt.xlabel(col)
        plt.ylabel('Density')
        
        # Add summary statistics to the plot
        stats_text = []
        for gender in df['genders'].unique():
            gender_data = df[df['genders'] == gender][col]
            stats_text.append(f'{gender}:\n'
                            f'Mean: {gender_data.mean():.3f}\n'
                            f'Std: {gender_data.std():.3f}\n'
                            f'Median: {gender_data.median():.3f}')
        
        plt.text(0.95, 0.95, '\n\n'.join(stats_text),
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.legend(title='Gender')
        plt.tight_layout()
        
        # Create safe filename
        safe_filename = create_safe_filename(col)
        plt.savefig(f'{output_dir}/distribution_{safe_filename}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a combined visualization of all distributions
    plt.figure(figsize=(15, 10))
    
    for idx, (col, title) in enumerate(measurements.items(), 1):
        plt.subplot(2, 3, idx)
        sns.kdeplot(data=df, x=col, hue='genders', fill=True, alpha=0.5, palette=gender_colors)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_distributions_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution visualizations have been saved to the '{output_dir}' directory.")

def create_combined_measurement_distribution(df, output_dir='sports_visualizations'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set the style for all plots
    plt.style.use('default')
    
    # Set a consistent figure size and font size
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['font.size'] = 10
    
    # Define the measurements to plot (order should be the same as in previous visualizations)
    measurements = [
        'Height (m)', 'Chest Circumference (m)', 'Waist Circumference (m)', 
        'Hip Circumference (m)', 'Mass (kg)', 'BMI ( kg / (m^2) )'
    ]
    
    # Check the unique values in the 'genders' column to adjust the color palette
    gender_values = df['genders'].unique()
    print(f"Unique gender values in the dataset: {gender_values}")
    
    # Adjust the color palette based on the actual gender values
    if 'Female' in gender_values and 'Male' in gender_values:
        color_palette = {"Female": "red", "Male": "blue"}
    else:
        # If genders are represented as 'F' and 'M' or other values, adjust accordingly
        color_palette = {"F": "red", "M": "blue"}
    
    # Create subplots to display all measurements in one figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Iterate through the measurements and plot them in individual subplots
    for idx, measurement in enumerate(measurements):
        ax = axes[idx]
        
        # Plot histogram for each gender separately with distinct colors
        sns.histplot(data=df, x=measurement, hue='genders', kde=True, 
                     bins=20, palette=color_palette, ax=ax, stat='density', multiple='layer')
        
        # Set the title and labels for each subplot
        ax.set_title(f'{measurement} Distribution by Gender')
        ax.set_xlabel(measurement)
        ax.set_ylabel('Density')
    
    # Adjust the layout of the subplots
    plt.tight_layout()
    
    # Create a safe filename for the plot
    safe_filename = create_safe_filename('combined_measurement_distribution')
    
    # Save the figure
    plt.savefig(f'{output_dir}/{safe_filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined measurement distribution plot has been saved to the '{output_dir}' directory.")

# Assuming df is already loaded with the required data
df = pd.read_csv('ssp3d.csv')
create_combined_measurement_distribution(df)
# Assuming df is already loaded with the required data
df = pd.read_csv('ssp3d.csv')
create_combined_measurement_distribution(df)