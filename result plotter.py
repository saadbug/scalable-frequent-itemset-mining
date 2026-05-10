import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- IEEE Styling Configuration ---
# Use standard serif fonts to match IEEE LaTeX templates
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300, # High resolution for publication
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_plots():
    if not os.path.exists("results.csv"):
        print("Error: results.csv not found. Please run benchmark.py first.")
        return

    # Load data
    df = pd.read_csv("results.csv")
    datasets = df['Dataset'].unique()
    
    # Create an output directory for the images
    os.makedirs("graphs", exist_ok=True)
    print("Generating IEEE-formatted graphs...")

    # Set Seaborn theme for clean academic look
    sns.set_theme(style="whitegrid", rc={"font.family":"serif"})

    for ds in datasets:
        ds_data = df[df['Dataset'] == ds].copy()
        
        # Sort values so the X-axis goes from High Support to Low Support (Left to Right)
        ds_data.sort_values(by='min_sup', ascending=False, inplace=True)
        # Convert min_sup to percentage string for cleaner X-axis
        ds_data['min_sup_str'] = (ds_data['min_sup'] * 100).astype(int).astype(str) + '%'

        # ==========================================
        # 1. Execution Time vs. Minimum Support
        # ==========================================
        plt.figure(figsize=(6, 4))
        sns.lineplot(
            data=ds_data, 
            x='min_sup_str', 
            y='Avg_Time_s', 
            hue='Algorithm', 
            style='Algorithm',
            markers=['o', 's'], # Circle for Apriori, Square for PVM
            dashes=False,
            linewidth=2,
            markersize=8,
            palette=['#d62728', '#1f77b4'] # Red for Apriori, Blue for PVM
        )
        
        plt.title(f'Execution Time vs. Minimum Support ({ds})')
        plt.xlabel('Minimum Support Threshold')
        plt.ylabel('Execution Time (Seconds) - Log Scale')
        plt.yscale('log') # Log scale because Apriori explodes exponentially
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(title='')
        
        time_file = f"graphs/time_{ds.lower()}.png"
        plt.savefig(time_file)
        plt.close()
        print(f" Saved: {time_file}")

        # ==========================================
        # 2. Memory Consumption Comparison
        # ==========================================
        plt.figure(figsize=(6, 4))
        sns.barplot(
            data=ds_data,
            x='min_sup_str',
            y='Avg_Memory_MB',
            hue='Algorithm',
            palette=['#d62728', '#1f77b4'],
            alpha=0.8
        )
        
        plt.title(f'Peak Memory Consumption ({ds})')
        plt.xlabel('Minimum Support Threshold')
        plt.ylabel('Memory Usage (MB)')
        plt.grid(axis='y', ls="--", alpha=0.5)
        plt.legend(title='')
        
        mem_file = f"graphs/memory_{ds.lower()}.png"
        plt.savefig(mem_file)
        plt.close()
        print(f" Saved: {mem_file}")

   # ==========================================
    # 3. Overall Speedup Ratio Graph
    # ==========================================
    plt.figure(figsize=(7, 4.5))
    
    # Filter only PVM rows since Speedup is relative to Apriori
    pvm_data = df[df['Algorithm'] == 'Parallel-Vertical-Miner'].copy()
    
    # Use a numerical column for the X-axis so it plots on a continuous mathematical scale
    pvm_data['min_sup_percent'] = pvm_data['min_sup'] * 100
    
    sns.lineplot(
        data=pvm_data,
        x='min_sup_percent',
        y='Speedup',
        hue='Dataset',
        style='Dataset',
        markers=['o', 's', '^'],
        dashes=False,
        linewidth=2,
        markersize=9,
        palette="viridis"
    )
    
    # Add a dashed line at Y=1 (Break-even point with Apriori)
    plt.axhline(1, color='red', linestyle='--', alpha=0.7, label='Apriori Baseline (1x)')
    
    plt.title('Parallel Vertical Miner Speedup vs. Apriori')
    plt.xlabel('Minimum Support Threshold (%)')
    plt.ylabel('Speedup Multiplier (x times faster)')
    

    plt.gca().invert_xaxis()
    
    plt.grid(True, ls="--", alpha=0.5)
    
    # Fix legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title='Dataset', loc='upper right')
    
    speed_file = "graphs/overall_speedup.png"
    plt.savefig(speed_file)
    plt.close()
    print(f" Saved: {speed_file}")
    
    print("\nAll IEEE graphs successfully generated in the 'graphs/' folder!")

if __name__ == "__main__":
    create_plots()