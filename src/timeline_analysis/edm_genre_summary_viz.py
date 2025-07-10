import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

def create_summary_infographic():
    """Create a comprehensive infographic summarizing the findings"""
    
    fig = plt.figure(figsize=(16, 20))
    
    # Main title
    fig.text(0.5, 0.98, 'THE PHANTOM GENRE PROBLEM:', ha='center', 
            fontsize=28, weight='bold')
    fig.text(0.5, 0.96, 'How EDM Went From 4 to 32 Genres Through Commerce, Not Music', 
            ha='center', fontsize=20, style='italic')
    
    # Create grid layout
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1.5, 1.2, 1, 0.8], 
                         hspace=0.3, wspace=0.2,
                         top=0.94, bottom=0.02)
    
    # Panel 1: The 8x explosion
    ax1 = fig.add_subplot(gs[0, :])
    
    # Draw arrow showing increase
    arrow_y = 0.5
    ax1.annotate('', xy=(0.85, arrow_y), xytext=(0.15, arrow_y),
                arrowprops=dict(arrowstyle='->', lw=8, color='darkred'))
    
    # 2004 box
    box_2004 = FancyBboxPatch((0.05, 0.3), 0.15, 0.4,
                             boxstyle="round,pad=0.02",
                             facecolor='#2c3e50', edgecolor='black', linewidth=3)
    ax1.add_patch(box_2004)
    ax1.text(0.125, 0.5, '4\nGENRES\n(2004)', ha='center', va='center',
            color='white', fontsize=20, weight='bold')
    
    # 2024 box
    box_2024 = FancyBboxPatch((0.8, 0.3), 0.15, 0.4,
                             boxstyle="round,pad=0.02",
                             facecolor='#e74c3c', edgecolor='black', linewidth=3)
    ax1.add_patch(box_2024)
    ax1.text(0.875, 0.5, '32\nGENRES\n(2024)', ha='center', va='center',
            color='white', fontsize=20, weight='bold')
    
    # 8x label
    ax1.text(0.5, arrow_y + 0.15, '8X INCREASE', ha='center', va='bottom',
            fontsize=24, weight='bold', color='darkred')
    ax1.text(0.5, arrow_y - 0.15, '20 YEARS', ha='center', va='top',
            fontsize=16, weight='bold', color='gray')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Panel 2: Commercial drivers wheel
    ax2 = fig.add_subplot(gs[1, :])
    
    # Center circle
    center = Circle((0.5, 0.5), 0.15, facecolor='gold', edgecolor='black', linewidth=3)
    ax2.add_patch(center)
    ax2.text(0.5, 0.5, 'GENRE\nCREATION', ha='center', va='center',
            fontsize=16, weight='bold')
    
    # Drivers around the circle
    drivers = [
        ('Festival Programming', 0.5, 0.85, '#e74c3c'),
        ('Beatport Categories', 0.85, 0.65, '#3498db'),
        ('Spotify Algorithms', 0.85, 0.35, '#9b59b6'),
        ('DJ Booking Markets', 0.5, 0.15, '#2ecc71'),
        ('Regional Commodification', 0.15, 0.35, '#f39c12'),
        ('Marketing Segmentation', 0.15, 0.65, '#e67e22')
    ]
    
    for driver, x, y, color in drivers:
        # Connection line
        ax2.plot([0.5, x], [0.5, y], 'k-', alpha=0.3, linewidth=2)
        
        # Driver circle
        driver_circle = Circle((x, y), 0.08, facecolor=color, 
                              edgecolor='black', linewidth=2, alpha=0.8)
        ax2.add_patch(driver_circle)
        
        # Label
        ha = 'left' if x > 0.5 else 'right' if x < 0.5 else 'center'
        x_offset = 0.1 if x > 0.5 else -0.1 if x < 0.5 else 0
        ax2.text(x + x_offset, y, driver, ha=ha, va='center',
                fontsize=12, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color, linewidth=2))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Commercial Forces Driving Genre Creation', 
                 fontsize=18, weight='bold')
    ax2.axis('off')
    
    # Panel 3: Phantom pairs comparison
    ax3 = fig.add_subplot(gs[2, :])
    
    pairs = [
        ('Deep House', 'Organic House', '75%', '#e74c3c'),
        ('Tech House', 'Minimal-Deep Tech', '79%', '#3498db'),
        ('Techno', 'Techno (Mood Labels)', '61%', '#2ecc71'),
        ('Trance', 'Trance (Venue Labels)', '73%', '#9b59b6')
    ]
    
    y_start = 0.85
    for i, (orig, phantom, sim, color) in enumerate(pairs):
        y = y_start - i * 0.22
        
        # Original
        orig_box = Rectangle((0.1, y-0.06), 0.25, 0.12, 
                           facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax3.add_patch(orig_box)
        ax3.text(0.225, y, orig, ha='center', va='center',
                fontsize=11, weight='bold', color='white')
        
        # Arrow
        ax3.annotate('', xy=(0.38, y), xytext=(0.35, y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Phantom
        phantom_box = Rectangle((0.4, y-0.06), 0.25, 0.12,
                              facecolor=color, edgecolor='black', 
                              linewidth=2, alpha=0.4, linestyle='--')
        ax3.add_patch(phantom_box)
        ax3.text(0.525, y, phantom, ha='center', va='center',
                fontsize=11, style='italic')
        
        # Similarity
        ax3.text(0.7, y, f'{sim} SIMILAR', ha='left', va='center',
                fontsize=14, weight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Our Case Studies: Phantom Genres with High Acoustic Overlap', 
                 fontsize=16, weight='bold')
    ax3.axis('off')
    
    # Panel 4: Money correlation
    ax4 = fig.add_subplot(gs[3, 0])
    
    # Data
    years = [2004, 2012, 2020, 2024]
    genres = [4, 12, 28, 32]
    revenue = [1.5, 4.5, 9.8, 11.8]
    
    # Normalize for visualization
    genres_norm = np.array(genres) / max(genres)
    revenue_norm = np.array(revenue) / max(revenue)
    
    # Plot bars
    x = np.arange(len(years))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, genres_norm, width, label='Genres (normalized)',
                    color='#3498db', edgecolor='black', linewidth=2)
    bars2 = ax4.bar(x + width/2, revenue_norm, width, label='Revenue (normalized)',
                    color='#2ecc71', edgecolor='black', linewidth=2)
    
    # Add actual values
    for i, (g, r) in enumerate(zip(genres, revenue)):
        ax4.text(i - width/2, genres_norm[i] + 0.02, str(g), 
                ha='center', va='bottom', fontsize=10, weight='bold')
        ax4.text(i + width/2, revenue_norm[i] + 0.02, f'${r}B', 
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(years)
    ax4.set_ylabel('Normalized Scale', fontsize=12)
    ax4.set_title('Perfect Correlation:\nMore Genres = More Money', 
                 fontsize=14, weight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Panel 5: Key quotes
    ax5 = fig.add_subplot(gs[3, 1])
    
    quotes = [
        "With 25,000 new releases\nweekly, genre tagging is\ncrucial for retail structure",
        "We conduct audio surveys\nto test new genres with DJs",
        "This sound was scattered\nover 3-4 genres, making it\ndifficult for customers"
    ]
    
    sources = ["Beatport VP", "Beatport Strategy", "Genre Creation Rationale"]
    
    y = 0.85
    for quote, source in zip(quotes, sources):
        # Quote box
        quote_box = FancyBboxPatch((0.05, y-0.15), 0.9, 0.2,
                                  boxstyle="round,pad=0.02",
                                  facecolor='lightgray', 
                                  edgecolor='black',
                                  linewidth=1, alpha=0.5)
        ax5.add_patch(quote_box)
        ax5.text(0.5, y-0.05, f'"{quote}"', ha='center', va='center',
                fontsize=10, style='italic')
        ax5.text(0.5, y-0.2, f'- {source}', ha='center', va='top',
                fontsize=9, weight='bold')
        y -= 0.3
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title('Industry Admits:\nGenres Created for Commerce', 
                 fontsize=14, weight='bold')
    ax5.axis('off')
    
    # Panel 6: Conclusion
    ax6 = fig.add_subplot(gs[4, :])
    
    conclusion_text = (
        "THE EVIDENCE IS CLEAR:\n\n"
        "• 8x genre increase driven by commercial platforms, not musical innovation\n"
        "• Phantom genres show 61-79% acoustic similarity - they're the same music\n"
        "• Industry openly admits creating genres for sales, playlists, and bookings\n"
        "• Each new category = new revenue stream (charts, playlists, festivals)\n\n"
        "Electronic music's 32-genre taxonomy is a commercial construct,\n"
        "not a reflection of genuine musical diversity."
    )
    
    conclusion_box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                   boxstyle="round,pad=0.03",
                                   facecolor='lightyellow', 
                                   edgecolor='darkred',
                                   linewidth=3)
    ax6.add_patch(conclusion_box)
    ax6.text(0.5, 0.5, conclusion_text, ha='center', va='center',
            fontsize=13, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('edm_genre_infographic.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate visualization
if __name__ == "__main__":
    create_summary_infographic()
    print("✓ Created comprehensive EDM genre infographic")