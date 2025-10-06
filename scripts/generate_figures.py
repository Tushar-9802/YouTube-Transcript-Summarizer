# save as: generate_paper_figures.py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Windows-compatible paths
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 300
sns.set_palette("husl")

# Create output directory
output_dir = Path('paper/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FIGURE 1: Results Comparison
# ============================================================================
domains = ['Medical', 'Engineering', 'Scientific']
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']

base_data = {
    'Medical': [0.341, 0.118, 0.294, 0.881],
    'Engineering': [0.347, 0.120, 0.301, 0.879],
    'Scientific': [0.336, 0.115, 0.290, 0.877]
}

lora_data = {
    'Medical': [0.283, 0.066, 0.263, 0.851],
    'Engineering': [0.272, 0.060, 0.254, 0.845],
    'Scientific': [0.288, 0.073, 0.265, 0.845]
}

base_std = {
    'Medical': [0.089, 0.056, 0.081, 0.023],
    'Engineering': [0.093, 0.061, 0.085, 0.026],
    'Scientific': [0.086, 0.054, 0.078, 0.025]
}

lora_std = {
    'Medical': [0.094, 0.045, 0.084, 0.028],
    'Engineering': [0.091, 0.038, 0.079, 0.031],
    'Scientific': [0.089, 0.042, 0.081, 0.029]
}

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

for idx, domain in enumerate(domains):
    ax = axes[idx]
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_data[domain], width, label='Base Model',
                   yerr=base_std[domain], capsize=3, alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, lora_data[domain], width, label='LoRA Model',
                   yerr=lora_std[domain], capsize=3, alpha=0.8, color='#A23B72')
    
    ax.set_ylabel('Score')
    ax.set_title(f'{domain} (n={[16,15,17][idx]})', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'results_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated: results_comparison.png")

# ============================================================================
# FIGURE 2: Training Loss from Actual CSVs
# ============================================================================
fig, ax = plt.subplots(figsize=(6, 3.5))

colors = {'medical': '#E63946', 'engineering': '#457B9D', 'scientific': '#2A9D8F'}
labels = {'medical': 'Medical', 'engineering': 'Engineering', 'scientific': 'Scientific'}

for domain in ['medical', 'engineering', 'scientific']:
    try:
        csv_path = Path(f'models/adapters/{domain}/training_log.csv')
        df = pd.read_csv(csv_path)
        
        # Plot loss vs epoch
        ax.plot(df['epoch'], df['loss'], label=labels[domain], 
                linewidth=2, color=colors[domain], alpha=0.85)
        
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found, skipping {domain}")
    except Exception as e:
        print(f"Error reading {domain}: {e}")

ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Training Loss', fontsize=11)
ax.set_title('LoRA Training Convergence', fontweight='bold', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(left=0)
plt.tight_layout()

plt.savefig(output_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated: training_loss.png")

# ============================================================================
# FIGURE 3: Error Distribution
# ============================================================================
categories = ['Academic\nFraming', 'Citation\nArtifacts', 'Style\nMismatch', 
              'Too Short', 'Too Verbose']

lora_freq = [14.6, 2.1, 54.0, 18.0, 13.0]
base_freq = [4.2, 0.0, 20.0, 12.0, 8.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, base_freq, width, label='Base Model',
                alpha=0.8, color='#2E86AB')
bars2 = ax1.bar(x + width/2, lora_freq, width, label='LoRA Model',
                alpha=0.8, color='#A23B72')

ax1.set_ylabel('Frequency (%)', fontsize=11)
ax1.set_title('Error Frequencies by Model', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=9)
ax1.legend()
ax1.grid(axis='y', alpha=0.3, linestyle='--')

differences = [l - b for l, b in zip(lora_freq, base_freq)]
colors_diff = ['#E63946' if d > 5 else '#F4A261' for d in differences]

bars = ax2.barh(categories, differences, alpha=0.8, color=colors_diff)
ax2.set_xlabel('LoRA Increase (%)', fontsize=11)
ax2.set_title('LoRA-Specific Error Increase', fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated: error_distribution.png")

# ============================================================================
# FIGURE 4: System Architecture
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
ax.axis('off')

boxes = [
    {'text': 'YouTube URL\nInput', 'pos': (0.5, 0.9), 'color': '#E8F4F8'},
    {'text': 'yt-dlp\nDownload', 'pos': (0.5, 0.75), 'color': '#B8E0D2'},
    {'text': 'Whisper\nTranscription\n(anti-hallucination)', 'pos': (0.5, 0.6), 'color': '#B8E0D2'},
    {'text': 'Mistral-7B\n(4-bit quantized)\n+ LoRA Adapter', 'pos': (0.5, 0.4), 'color': '#D6EADF'},
    {'text': 'Post-Processing\n(keywords, highlighting)', 'pos': (0.5, 0.25), 'color': '#B8E0D2'},
    {'text': 'Summary\nOutput', 'pos': (0.5, 0.1), 'color': '#E8F4F8'},
]

for box in boxes:
    bbox = dict(boxstyle='round,pad=0.5', facecolor=box['color'], 
                edgecolor='black', linewidth=2)
    ax.text(box['pos'][0], box['pos'][1], box['text'], 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=bbox, transform=ax.transAxes)

arrow_props = dict(arrowstyle='->', lw=2, color='black')
for i in range(len(boxes) - 1):
    ax.annotate('', xy=(boxes[i+1]['pos'][0], boxes[i+1]['pos'][1] + 0.05),
                xytext=(boxes[i]['pos'][0], boxes[i]['pos'][1] - 0.05),
                arrowprops=arrow_props, transform=ax.transAxes)

ax.text(0.85, 0.4, '8GB VRAM\n~6.2GB peak', ha='center', va='center',
        fontsize=8, style='italic', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax.text(0.15, 0.6, 'Dual ASR:\nPyTorch Whisper\nFaster-Whisper', 
        ha='center', va='center', fontsize=7, style='italic',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(output_dir / 'system_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated: system_architecture.png")

print("\n" + "="*60)
print("All figures saved to: paper/figures/")
print("="*60)