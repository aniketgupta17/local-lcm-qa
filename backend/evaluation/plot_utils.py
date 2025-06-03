import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

def plot_metrics_by_domain(metrics, save_dir, timestamp):
    """Generate publication-quality plots for all metrics."""
    os.makedirs(os.path.join(save_dir, f"plots_{timestamp}"), exist_ok=True)
    plot_dir = os.path.join(save_dir, f"plots_{timestamp}")
    
    # If only latency is present, plot only latency
    if list(metrics.keys()) == ['latency']:
        plot_latency_visualizations(metrics['latency'], plot_dir, timestamp)
        return

    # Helper to check if a metric is non-zero for any domain
    def nonzero_metric(metric_dict):
        return any(v > 0 for k, v in metric_dict.items() if k != 'overall')

    metric_keys = [
        ('hallucination_scores', 'Hallucination Score'),
        ('confidence_calibration', 'Confidence Calibration'),
        ('source_overlap', 'Source Grounding'),
        ('trace_scores', 'TRACe Score'),
        ('exact_match', 'Exact Match'),
        ('precision@k', 'Precision@k')
    ]

    for key, label in metric_keys:
        if key in metrics and nonzero_metric(metrics[key]):
            plot_metric_visualizations(key, label, metrics[key], plot_dir, timestamp)
    
    # Always plot latency
    plot_latency_visualizations(metrics['latency'], plot_dir, timestamp)
    
    # Create comprehensive dashboard for all metrics
    create_metrics_dashboard(metrics, plot_dir, timestamp)

def plot_metric_visualizations(key, label, metric_data, save_dir, timestamp):
    """Generate multiple visualization types for a given metric."""
    domains = [d for d in metric_data.keys() if d != 'overall']
    values = [metric_data[d] for d in domains]
    
    # 1. Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=domains, y=values, ax=ax)
    ax.set_title(f'{label} by Domain')
    ax.set_ylabel(label)
    ax.set_xlabel('Domain')
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{key}_bar_{timestamp}.png'))
    plt.close()
    
    # 2. Horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(y=domains, x=values, ax=ax, orient='h')
    ax.set_title(f'{label} by Domain (Horizontal)')
    ax.set_xlabel(label)
    ax.set_ylabel('Domain')
    for i, v in enumerate(values):
        ax.text(v, i, f' {v:.2f}', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{key}_hbar_{timestamp}.png'))
    plt.close()
    
    # 3. Line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=domains, y=values, marker='o', ax=ax, linewidth=2)
    ax.set_title(f'{label} Trend Across Domains')
    ax.set_ylabel(label)
    ax.set_xlabel('Domain')
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{key}_line_{timestamp}.png'))
    plt.close()
    
    # 4. Pie chart (if sum > 0)
    if sum(values) > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        explode = [0.05] * len(domains)  # explode all slices a bit
        ax.pie(values, labels=domains, autopct='%1.1f%%', startangle=140, explode=explode,
               shadow=True, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        ax.set_title(f'{label} Distribution by Domain')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{key}_pie_{timestamp}.png'))
        plt.close()
    
    # 5. Polar chart
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    angles = np.linspace(0, 2*np.pi, len(domains), endpoint=False).tolist()
    # Close the loop
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]
    ax.plot(angles_closed, values_closed, 'o-', linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(domains)
    ax.set_yticks([])  # Hide radial ticks
    ax.set_title(f'{label} Polar Plot')
    # Add value labels
    for angle, value, domain in zip(angles, values, domains):
        ax.text(angle, value, f'{value:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{key}_polar_{timestamp}.png'))
    plt.close()
    
    # 6. Heatmap with single row
    fig, ax = plt.subplots(figsize=(10, 3))
    data = pd.DataFrame([values], columns=domains)
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)
    ax.set_title(f'{label} Intensity by Domain')
    ax.set_xticklabels(domains)
    ax.set_yticklabels(['Score'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{key}_heatmap_{timestamp}.png'))
    plt.close()

def plot_latency_visualizations(latency_data, save_dir, timestamp):
    """Generate multiple visualization types for latency metrics."""
    metrics_order = ['mean', 'median', 'p90', 'p99']
    latency_values = [latency_data[m] for m in metrics_order]
    
    # 1. Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=metrics_order, y=latency_values, ax=ax)
    ax.set_title('Latency Distribution with Percentiles')
    ax.set_ylabel('Latency (seconds)')
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(latency_values):
        ax.text(i, v, f'{v:.2f}s', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'latency_bar_{timestamp}.png'))
    plt.close()
    
    # 2. Horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(y=metrics_order, x=latency_values, ax=ax, orient='h')
    ax.set_title('Latency Distribution (Horizontal)')
    ax.set_xlabel('Latency (seconds)')
    for i, v in enumerate(latency_values):
        ax.text(v, i, f' {v:.2f}s', va='center')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'latency_hbar_{timestamp}.png'))
    plt.close()
    
    # 3. Line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=metrics_order, y=latency_values, marker='o', ax=ax, linewidth=2)
    ax.set_title('Latency Metrics Trend')
    ax.set_ylabel('Latency (seconds)')
    for i, v in enumerate(latency_values):
        ax.text(i, v, f'{v:.2f}s', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'latency_line_{timestamp}.png'))
    plt.close()
    
    # 4. Area chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(metrics_order, latency_values, alpha=0.5)
    ax.plot(metrics_order, latency_values, 'o-', linewidth=2)
    ax.set_title('Latency Area Plot')
    ax.set_ylabel('Latency (seconds)')
    for i, v in enumerate(latency_values):
        ax.text(i, v, f'{v:.2f}s', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'latency_area_{timestamp}.png'))
    plt.close()
    
    # 5. Gauge chart (simplified as half-circle)
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'polar'})
    # Convert to half circle
    theta = np.linspace(0, np.pi, 100)
    # Normalize mean latency relative to p99 for gauge
    r = np.ones_like(theta) * (latency_data['mean'] / latency_data['p99'])
    ax.plot(theta, r)
    ax.fill(theta, r, alpha=0.5)
    ax.set_rticks([])  # Hide radial ticks
    ax.set_xticks([])  # Hide angular ticks
    ax.set_title(f'Latency Gauge: Mean {latency_data["mean"]:.2f}s / P99 {latency_data["p99"]:.2f}s')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'latency_gauge_{timestamp}.png'))
    plt.close()

def create_metrics_dashboard(metrics, save_dir, timestamp):
    """Create a comprehensive dashboard with all metrics."""
    # Only include metrics that have non-zero values
    nonzero_metrics = {}
    for key in metrics:
        if key != 'latency':
            if 'overall' in metrics[key] and metrics[key]['overall'] > 0:
                nonzero_metrics[key] = metrics[key]['overall']
    
    if not nonzero_metrics:
        return  # No non-zero metrics to display
    
    # Create a dashboard with radar chart and bar charts
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Radar chart for overall metrics
    ax_radar = fig.add_subplot(gs[0, 0], projection='polar')
    categories = list(nonzero_metrics.keys())
    values = [nonzero_metrics[cat] for cat in categories]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    # Close the loop
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]
    ax_radar.plot(angles_closed, values_closed, 'o-', linewidth=2)
    ax_radar.fill(angles_closed, values_closed, alpha=0.25)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels([k.replace('_scores', '').replace('_', ' ').title() for k in categories])
    ax_radar.set_title('Overall System Performance')
    
    # Bar chart for overall metrics
    ax_bar = fig.add_subplot(gs[0, 1])
    sns.barplot(x=list(nonzero_metrics.keys()), y=list(nonzero_metrics.values()), ax=ax_bar)
    ax_bar.set_title('Overall Metrics Comparison')
    ax_bar.set_xticklabels([k.replace('_scores', '').replace('_', ' ').title() for k in nonzero_metrics.keys()], rotation=45)
    ax_bar.set_ylabel('Score')
    for i, v in enumerate(nonzero_metrics.values()):
        ax_bar.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Latency bar chart
    ax_latency = fig.add_subplot(gs[1, :])
    metrics_order = ['mean', 'median', 'p90', 'p99']
    latency_values = [metrics['latency'][m] for m in metrics_order]
    sns.barplot(x=metrics_order, y=latency_values, ax=ax_latency)
    ax_latency.set_title('Latency Distribution')
    ax_latency.set_ylabel('Latency (seconds)')
    for i, v in enumerate(latency_values):
        ax_latency.text(i, v, f'{v:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'metrics_dashboard_{timestamp}.png'))
    plt.close()

def bar_chart(df, x, y, title, save_path):
    """Create a publication-quality bar chart."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def box_plot(df, x, y, title, save_path):
    """Create a publication-quality box plot."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def line_chart(df, x, y, title, save_path):
    """Create a publication-quality line chart."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x, y=y, marker='o')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def radar_chart(categories, values, title, save_path):
    """Create a publication-quality radar chart."""
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close() 