#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import os

# Set the style for beautiful plots
plt.style.use('seaborn-v0_8-pastel')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10

# Create custom color palettes
vibrant_colors = ['#1a53ff', '#ff5e5e', '#00cc96', '#ab63fa', '#ffa15a', '#19d3f3', '#ff6692', '#b6e880']
pastel_colors = ['#a8c3ff', '#ffb3b3', '#b3ffda', '#e0c3ff', '#ffd9b3', '#b3f0ff', '#ffb3d9', '#deffb3']
gradient_cmap = LinearSegmentedColormap.from_list('custom_gradient', ['#4b0082', '#9370db', '#e6e6fa'])
diverging_cmap = LinearSegmentedColormap.from_list('custom_diverging', ['#d73027', '#ffffbf', '#1a9850'])

print("Generating comprehensive visualizations for thesis results...")

# Load the cleaned JSON files
with open('cleaned_without_pipeline.json', 'r') as f:
    without_pipeline = json.load(f)

with open('cleaned_with_pipeline.json', 'r') as f:
    with_pipeline = json.load(f)

# Extract metrics from the JSON data
without_metrics = without_pipeline['metrics']
with_metrics = with_pipeline['metrics']

# Get the list of domains from the domain_performance section
all_domains = list(without_pipeline['domain_performance'].keys())

# Calculate improvements
improvements = {}
for metric_name in with_metrics:
    if metric_name in without_metrics and metric_name != 'latency' and metric_name != 'aggregate_scores':
        improvements[metric_name] = {}
        
        # Calculate overall improvement
        if 'overall' in with_metrics[metric_name] and 'overall' in without_metrics[metric_name]:
            original = without_metrics[metric_name]['overall']
            improved = with_metrics[metric_name]['overall']
            
            # Avoid division by zero
            if original > 0:
                improvements[metric_name]['overall'] = ((improved - original) / original) * 100
            else:
                improvements[metric_name]['overall'] = 0 if improved == 0 else 100
        
        # Calculate domain-specific improvements
        for domain in all_domains:
            if (domain in with_metrics[metric_name] and 
                domain in without_metrics[metric_name]):
                original = without_metrics[metric_name][domain]
                improved = with_metrics[metric_name][domain]
                
                # Avoid division by zero
                if original > 0:
                    improvements[metric_name][domain] = ((improved - original) / original) * 100
                else:
                    improvements[metric_name][domain] = 0 if improved == 0 else 100

# 1. Overall Performance Comparison
def plot_overall_performance():
    metrics_to_plot = ['exact_match', 'source_overlap', 'trace_scores', 'hallucination_scores']
    
    # Create dataframe for plotting
    data = []
    for metric in metrics_to_plot:
        if metric in without_metrics and metric in with_metrics:
            data.append({
                'Metric': metric.replace('_scores', '').replace('_', ' ').title(),
                'Without Pipeline': without_metrics[metric]['overall'],
                'With Pipeline': with_metrics[metric]['overall']
            })
    
    df = pd.DataFrame(data)
    
    # Melt the dataframe for easier plotting
    df_melted = pd.melt(df, id_vars=['Metric'], var_name='Method', value_name='Score')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Use a line plot with markers for better visualization
    ax = sns.lineplot(
        data=df_melted, 
        x='Metric', 
        y='Score', 
        hue='Method', 
        style='Method',
        markers=True, 
        dashes=False,
        palette=[vibrant_colors[0], vibrant_colors[2]]
    )
    
    # Add the actual data points with larger markers
    sns.scatterplot(
        data=df_melted, 
        x='Metric', 
        y='Score', 
        hue='Method',
        s=150,
        alpha=0.7,
        palette=[vibrant_colors[0], vibrant_colors[2]],
        legend=False
    )
    
    # Add data labels
    for i, row in df_melted.iterrows():
        ax.text(
            i % len(metrics_to_plot), 
            row['Score'], 
            f"{row['Score']:.2f}", 
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    # Style the plot
    plt.title('Performance Comparison: Standard vs. Enhanced Pipeline', fontsize=18, fontweight='bold')
    plt.ylabel('Score', fontsize=16)
    plt.ylim(bottom=0)  # Start y-axis at 0
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Method', title_fontsize=14, fontsize=12, loc='best')
    
    # Add a subtle background color
    ax.set_facecolor('#f8f8f8')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('overall_performance_comparison.png', bbox_inches='tight')
    plt.close()

# 2. Domain Performance Radar Chart
def plot_domain_radar():
    # Select a metric to visualize across domains
    metric = 'trace_scores'  # Change as needed
    
    if metric in with_metrics and metric in without_metrics:
        # Create data for the radar chart
        domains = []
        without_values = []
        with_values = []
        
        for domain in all_domains:
            if domain in with_metrics[metric] and domain in without_metrics[metric]:
                domains.append(domain)
                without_values.append(without_metrics[metric][domain])
                with_values.append(with_metrics[metric][domain])
        
        # Create radar chart
        fig = plt.figure(figsize=(12, 10))
        
        # Calculate angles for each domain
        angles = np.linspace(0, 2*np.pi, len(domains), endpoint=False).tolist()
        
        # Close the loop
        domains = domains + [domains[0]]
        without_values = without_values + [without_values[0]]
        with_values = with_values + [with_values[0]]
        angles = angles + [angles[0]]
        
        # Create subplot with polar projection
        ax = fig.add_subplot(111, polar=True)
        
        # Plot without pipeline data
        ax.plot(angles, without_values, 'o-', linewidth=2.5, label='Without Pipeline', color=vibrant_colors[0])
        ax.fill(angles, without_values, alpha=0.2, color=vibrant_colors[0])
        
        # Plot with pipeline data
        ax.plot(angles, with_values, 'o-', linewidth=2.5, label='With Pipeline', color=vibrant_colors[2])
        ax.fill(angles, with_values, alpha=0.2, color=vibrant_colors[2])
        
        # Set ticks and labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(domains[:-1], fontsize=12)
        
        # Add metric values as text
        for i in range(len(domains)-1):
            ax.text(angles[i], without_values[i], f"{without_values[i]:.2f}", 
                    ha='center', va='bottom', color=vibrant_colors[0], fontweight='bold')
            ax.text(angles[i], with_values[i], f"{with_values[i]:.2f}", 
                    ha='center', va='bottom', color=vibrant_colors[2], fontweight='bold')
        
        # Style the plot
        ax.set_facecolor('#f8f8f8')
        plt.title(f'Domain-Specific Performance: {metric.replace("_scores", "").replace("_", " ").title()}', 
                  fontsize=18, fontweight='bold', y=1.08)
        plt.legend(loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('domain_performance_radar.png', bbox_inches='tight')
        plt.close()

# 3. Latency and Improvements Analysis
def plot_latency_and_improvements():
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Latency comparison
    latency_data = {
        'Method': ['Without Pipeline', 'With Pipeline'],
        'Mean': [without_metrics.get('latency', {}).get('mean', 0), 
                 with_metrics.get('latency', {}).get('mean', 0)],
        'Median': [without_metrics.get('latency', {}).get('median', 0), 
                   with_metrics.get('latency', {}).get('median', 0)],
        'P90': [without_metrics.get('latency', {}).get('p90', 0), 
                with_metrics.get('latency', {}).get('p90', 0)],
        'P99': [without_metrics.get('latency', {}).get('p99', 0), 
                with_metrics.get('latency', {}).get('p99', 0)]
    }
    
    latency_df = pd.DataFrame(latency_data)
    latency_melted = pd.melt(latency_df, id_vars=['Method'], var_name='Metric', value_name='Seconds')
    
    # Create line plot for latency
    sns.lineplot(
        data=latency_melted, 
        x='Metric', 
        y='Seconds', 
        hue='Method', 
        style='Method',
        markers=True, 
        dashes=False,
        palette=[vibrant_colors[0], vibrant_colors[2]],
        ax=ax1
    )
    
    # Add scatter points for emphasis
    sns.scatterplot(
        data=latency_melted, 
        x='Metric', 
        y='Seconds', 
        hue='Method',
        s=150,
        alpha=0.7,
        palette=[vibrant_colors[0], vibrant_colors[2]],
        ax=ax1,
        legend=False
    )
    
    # Add data labels
    for i, row in latency_melted.iterrows():
        ax1.text(
            i % 4, 
            row['Seconds'], 
            f"{row['Seconds']:.2f}s", 
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    # Style the first plot
    ax1.set_title('Latency Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Seconds', fontsize=14)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(title='Method', fontsize=12)
    ax1.set_facecolor('#f8f8f8')
    
    # Plot 2: Improvements for selected metrics
    metrics_to_plot = ['exact_match', 'source_overlap', 'trace_scores']
    improvement_data = []
    
    for metric in metrics_to_plot:
        if metric in improvements and 'overall' in improvements[metric]:
            improvement_pct = improvements[metric]['overall']
            improvement_data.append({
                'Metric': metric.replace('_scores', '').replace('_', ' ').title(),
                'Improvement (%)': improvement_pct
            })
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Create a horizontal bar chart for improvements
    bars = sns.barplot(
        y='Metric', 
        x='Improvement (%)', 
        data=improvement_df,
        palette=sns.color_palette(vibrant_colors[:len(metrics_to_plot)]),
        ax=ax2
    )
    
    # Add data labels
    for i, v in enumerate(improvement_df['Improvement (%)']):
        ax2.text(
            v + 1, 
            i, 
            f"{v:.1f}%", 
            va='center',
            fontweight='bold'
        )
    
    # Style the second plot
    ax2.set_title('Performance Improvements (%)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Improvement (%)', fontsize=14)
    ax2.set_xlim(left=0)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#f8f8f8')
    
    plt.tight_layout()
    plt.savefig('latency_and_improvements.png', bbox_inches='tight')
    plt.close()

# 4. Detailed Metrics Heatmap
def plot_detailed_heatmap():
    # Create a dataframe for the heatmap
    heatmap_data = []
    
    for metric in improvements:
        for domain in all_domains:
            if domain in improvements[metric]:
                heatmap_data.append({
                    'Metric': metric.replace('_scores', '').replace('_', ' ').title(),
                    'Domain': domain,
                    'Improvement (%)': improvements[metric][domain]
                })
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Pivot the dataframe for the heatmap
        heatmap_pivot = heatmap_df.pivot(index='Metric', columns='Domain', values='Improvement (%)')
        
        # Create the figure
        plt.figure(figsize=(14, 10))
        
        # Create the heatmap with improved colors
        ax = sns.heatmap(
            heatmap_pivot, 
            annot=True, 
            fmt='.1f', 
            cmap=diverging_cmap,
            center=0,
            linewidths=.5,
            cbar_kws={'label': 'Improvement (%)'}
        )
        
        # Style the heatmap
        plt.title('Detailed Performance Improvements by Domain and Metric', fontsize=18, fontweight='bold')
        plt.ylabel('Metric', fontsize=16)
        plt.xlabel('Domain', fontsize=16)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('detailed_metrics_heatmap.png', bbox_inches='tight')
        plt.close()

# Generate all visualizations
plot_overall_performance()
plot_domain_radar()
plot_latency_and_improvements()
plot_detailed_heatmap()

print("\nAll visualizations have been saved to the 'results2' folder:")
print("1. overall_performance_comparison.png - Key metrics comparison")
print("2. domain_performance_radar.png - Domain-specific performance radar")
print("3. latency_and_improvements.png - Latency and improvement analysis")
print("4. detailed_metrics_heatmap.png - Comprehensive improvement heatmap") 