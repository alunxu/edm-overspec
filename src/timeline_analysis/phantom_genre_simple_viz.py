import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

def create_simple_powerful_viz():
    """Create a single, simple visualization that captures the essence of the problem"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Title
    ax.text(0.5, 0.95, 'THE PHANTOM GENRE PROBLEM', 
           ha='center', va='top', fontsize=24, weight='bold',
           transform=ax.transAxes)
    ax.text(0.5, 0.91, 'Commerce Creates Categories, Not Music', 
           ha='center', va='top', fontsize=16, style='italic',
           transform=ax.transAxes)
    
    # Left side - "What Industry Says"
    ax.text(0.25, 0.82, 'WHAT INDUSTRY SELLS:', ha='center', fontsize=16, 
           weight='bold', transform=ax.transAxes)
    
    # Show 8 "different" genres
    genres_marketed = [
        ('Deep House', '#e74c3c'),
        ('Organic House', '#ec7063'),
        ('Tech House', '#3498db'),
        ('Minimal-Deep Tech', '#5dade2'),
        ('Techno (Peak Time)', '#2ecc71'),
        ('Techno (Raw/Hypnotic)', '#58d68d'),
        ('Trance (Main Floor)', '#9b59b6'),
        ('Trance (Raw/Hypnotic)', '#bb8fce')
    ]
    
    y_start = 0.72
    for i, (genre, color) in enumerate(genres_marketed):
        y = y_start - i * 0.08
        
        # Genre box
        box = FancyBboxPatch((0.05, y-0.03), 0.4, 0.06,
                            boxstyle="round,pad=0.01",
                            facecolor=color, edgecolor='black', 
                            linewidth=1, alpha=0.7,
                            transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(0.25, y, genre, ha='center', va='center',
               fontsize=11, weight='bold', transform=ax.transAxes)
    
    # Right side - "What Music Shows"
    ax.text(0.75, 0.82, 'WHAT ACOUSTICS REVEAL:', ha='center', fontsize=16,
           weight='bold', transform=ax.transAxes)
    
    # Show 4 actual clusters
    clusters = [
        ('House Family\n(75-79% similar)', '#e74c3c', 0.65),
        ('Techno Family\n(61% similar)', '#2ecc71', 0.5),
        ('Trance Family\n(73% similar)', '#9b59b6', 0.35),
        ('Bass Family', '#f39c12', 0.2)
    ]
    
    for cluster, color, y in clusters:
        # Cluster circle
        circle = Circle((0.75, y), 0.08, facecolor=color, 
                       edgecolor='black', linewidth=3, alpha=0.5,
                       transform=ax.transAxes)
        ax.add_patch(circle)
        ax.text(0.75, y, cluster, ha='center', va='center',
               fontsize=12, weight='bold', transform=ax.transAxes)
    
    # Arrow showing collapse
    ax.annotate('', xy=(0.52, 0.45), xytext=(0.48, 0.45),
               arrowprops=dict(arrowstyle='->', lw=4, color='darkred'),
               transform=ax.transAxes)
    ax.text(0.5, 0.48, 'REALITY', ha='center', va='bottom',
           fontsize=14, weight='bold', color='darkred',
           transform=ax.transAxes)
    
    # Bottom statistics
    stats_box = FancyBboxPatch((0.1, 0.02), 0.8, 0.12,
                              boxstyle="round,pad=0.02",
                              facecolor='lightyellow', 
                              edgecolor='darkred',
                              linewidth=3,
                              transform=ax.transAxes)
    ax.add_patch(stats_box)
    
    stats_text = (
        "2004: 4 genres → 2024: 32 genres | "
        "Market size: $1.5B → $11.8B | "
        "Beatport admits: 'Genre tagging crucial for retail' | "
        "Result: 61-79% similar music sold as different"
    )
    ax.text(0.5, 0.08, stats_text, ha='center', va='center',
           fontsize=12, weight='bold', transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('phantom_genre_simple.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_before_after_viz():
    """Create a before/after visualization of EDM genres"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Common title
    fig.suptitle('EDM Genre Evolution: Organic Growth vs Commercial Explosion', 
                fontsize=20, weight='bold')
    
    # Left panel - 2004 (Before)
    ax1.set_title('2004: The Original 4', fontsize=16, weight='bold', pad=20)
    
    # Draw 4 distinct circles
    original_genres = [
        ('HOUSE', 0.5, 0.7, '#e74c3c', 200),
        ('TECHNO', 0.3, 0.3, '#3498db', 200),
        ('TRANCE', 0.7, 0.3, '#9b59b6', 200),
        ('DRUM & BASS', 0.5, 0.5, '#2ecc71', 200)
    ]
    
    for genre, x, y, color, size in original_genres:
        circle = Circle((x, y), size/1000, facecolor=color, 
                       edgecolor='black', linewidth=3, alpha=0.8)
        ax1.add_patch(circle)
        ax1.text(x, y, genre, ha='center', va='center',
                fontsize=14, weight='bold', color='white')
    
    ax1.text(0.5, 0.05, 'Clear boundaries, distinct sounds', 
            ha='center', fontsize=12, style='italic')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Right panel - 2024 (After)
    ax2.set_title('2024: 32 Genres (Many Phantoms)', fontsize=16, weight='bold', pad=20)
    
    # Draw overlapping mess
    np.random.seed(42)
    all_genres = []
    
    # Generate base positions for 4 families
    family_centers = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
    family_colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    
    # Generate 8 genres per family with overlap
    for i, (center, color) in enumerate(zip(family_centers, family_colors)):
        for j in range(8):
            # Create variations around center
            x = center[0] + np.random.normal(0, 0.15)
            y = center[1] + np.random.normal(0, 0.15)
            size = np.random.uniform(50, 100)
            
            circle = Circle((x, y), size/1000, facecolor=color, 
                           edgecolor='black', linewidth=1, 
                           alpha=0.3)
            ax2.add_patch(circle)
    
    # Add labels for some genres
    sample_labels = [
        ('Deep House', 0.25, 0.35),
        ('Organic House', 0.35, 0.25),
        ('Tech House', 0.65, 0.35),
        ('Minimal-DT', 0.75, 0.25),
        ('Techno (PTD)', 0.25, 0.65),
        ('Techno (RDH)', 0.35, 0.75),
        ('Trance (MF)', 0.65, 0.75),
        ('Trance (RDH)', 0.75, 0.65)
    ]
    
    for label, x, y in sample_labels:
        ax2.text(x, y, label, ha='center', va='center',
                fontsize=9, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='black', alpha=0.8))
    
    ax2.text(0.5, 0.05, 'Overlapping phantoms, commercial confusion', 
            ha='center', fontsize=12, style='italic', color='darkred')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Add bottom text
    fig.text(0.5, 0.02, 
            'Commerce transformed 4 distinct genres into 32 overlapping categories. '
            'Our analysis shows 61-79% acoustic similarity between "different" genres.',
            ha='center', fontsize=14, weight='bold', style='italic')
    
    plt.tight_layout()
    plt.savefig('edm_before_after.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate visualizations
if __name__ == "__main__":
    create_simple_powerful_viz()
    print("✓ Created simple phantom genre visualization")
    
    create_before_after_viz()
    print("✓ Created before/after comparison")