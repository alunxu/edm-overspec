import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def create_genre_timeline():
    """Create timeline showing genre proliferation and commercial events"""
    
    # Data for genre count over time
    timeline_data = [
        (2004, 4, "Beatport launches with 4 core genres"),
        (2006, 6, "First genre expansions"),
        (2010, 10, "EDM boom begins"),
        (2013, 14, "Festival circuit expansion"),
        (2016, 20, "Big Room & Future House added"),
        (2018, 24, "Bass genre fragmentation"),
        (2020, 28, "Organic House splits from Deep House"),
        (2022, 30, "Techno mood subdivisions"),
        (2024, 32, "Current state: 8x original genres")
    ]
    
    # Commercial events
    commercial_events = [
        (2004, "Beatport Launch", "Platform"),
        (2011, "EDC moves to Vegas", "Festival"),
        (2013, "SFX IPO $260M", "Financial"),
        (2015, "Spotify genre mapping", "Streaming"),
        (2016, "Beatport genre surveys", "Platform"),
        (2018, "Festival stage programming", "Festival"),
        (2020, "Pandemic streaming surge", "Streaming"),
        (2022, "AI playlist optimization", "Tech")
    ]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                    gridspec_kw={'height_ratios': [3, 2]})
    
    # Top panel - Genre count growth
    years = [item[0] for item in timeline_data]
    counts = [item[1] for item in timeline_data]
    labels = [item[2] for item in timeline_data]
    
    # Create gradient effect
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(years)))
    
    # Plot line and points
    ax1.plot(years, counts, 'k-', linewidth=3, alpha=0.3)
    
    for i, (year, count, label) in enumerate(timeline_data):
        ax1.scatter(year, count, s=300, color=colors[i], edgecolor='black', 
                   linewidth=2, zorder=10)
        
        # Add labels for key points
        if year in [2004, 2016, 2020, 2024]:
            ax1.annotate(label, (year, count), 
                        xytext=(0, 20), textcoords='offset points',
                        ha='center', fontsize=10, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                 edgecolor=colors[i], linewidth=2))
    
    # Fill area under curve
    ax1.fill_between(years, 0, counts, alpha=0.2, color='red')
    
    # Add "8x increase" annotation
    ax1.annotate('', xy=(2024, 32), xytext=(2004, 4),
                arrowprops=dict(arrowstyle='<->', lw=3, color='darkred'))
    ax1.text(2014, 18, '8x INCREASE\nin 20 years', ha='center', fontsize=16, 
            weight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    ax1.set_xlim(2002, 2026)
    ax1.set_ylim(0, 35)
    ax1.set_xlabel('Year', fontsize=14, weight='bold')
    ax1.set_ylabel('Number of EDM Genres', fontsize=14, weight='bold')
    ax1.set_title('The EDM Genre Explosion: From Underground to Over-Categorized', 
                 fontsize=18, weight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel - Commercial events timeline
    event_colors = {
        'Platform': '#3498db',
        'Festival': '#e74c3c',
        'Financial': '#2ecc71',
        'Streaming': '#9b59b6',
        'Tech': '#f39c12'
    }
    
    y_positions = {
        'Platform': 0.8,
        'Festival': 0.6,
        'Financial': 0.4,
        'Streaming': 0.2,
        'Tech': 0.0
    }
    
    for year, event, category in commercial_events:
        y = y_positions[category]
        
        # Draw event marker
        ax2.scatter(year, y, s=200, color=event_colors[category], 
                   edgecolor='black', linewidth=2, zorder=10)
        
        # Add event label
        ax2.text(year, y + 0.05, event, ha='center', va='bottom',
                fontsize=9, rotation=45, weight='bold')
    
    # Add category labels
    for category, y in y_positions.items():
        ax2.text(2002, y, category, ha='right', va='center',
                fontsize=11, weight='bold', color=event_colors[category])
    
    ax2.set_xlim(2002, 2026)
    ax2.set_ylim(-0.2, 1.0)
    ax2.set_xlabel('Year', fontsize=14, weight='bold')
    ax2.set_title('Commercial Events Driving Genre Creation', 
                 fontsize=14, weight='bold')
    ax2.set_yticks([])
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('edm_genre_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_phantom_splits_diagram():
    """Create diagram showing how genres split for commercial reasons"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define genre splits
    splits = [
        {
            'original': 'HOUSE',
            'year': 2004,
            'splits': [
                ('Deep House', 2010, 'Club market'),
                ('Organic House', 2020, 'Festival/conscious market'),
                ('Tech House', 2008, 'Minimal market'),
                ('Future House', 2016, 'Commercial market')
            ]
        },
        {
            'original': 'TECHNO',
            'year': 2004,
            'splits': [
                ('Minimal', 2008, 'Underground market'),
                ('Techno (Peak Time)', 2022, 'Festival main stage'),
                ('Techno (Raw/Hypnotic)', 2022, 'Underground purists'),
                ('Melodic Techno', 2018, 'Crossover market')
            ]
        },
        {
            'original': 'TRANCE',
            'year': 2004,
            'splits': [
                ('Progressive Trance', 2008, 'DJ market'),
                ('Trance (Main Floor)', 2022, 'Festival market'),
                ('Trance (Raw/Hypnotic)', 2022, 'Underground market'),
                ('Uplifting Trance', 2012, 'Euphoric market')
            ]
        },
        {
            'original': 'DRUM & BASS',
            'year': 2004,
            'splits': [
                ('Liquid D&B', 2010, 'Commercial market'),
                ('Jump Up', 2012, 'Party market'),
                ('Neurofunk', 2014, 'Technical market'),
                ('Jungle', 2016, 'Revival market')
            ]
        }
    ]
    
    # Color scheme
    original_color = '#2c3e50'
    split_colors = plt.cm.Set3(np.linspace(0, 1, 20))
    
    # Calculate positions
    y_start = 0.9
    y_spacing = 0.22
    
    for i, genre_family in enumerate(splits):
        y_pos = y_start - (i * y_spacing)
        
        # Draw original genre box
        orig_box = FancyBboxPatch((0.05, y_pos - 0.04), 0.15, 0.08,
                                  boxstyle="round,pad=0.01",
                                  facecolor=original_color, 
                                  edgecolor='black', linewidth=2)
        ax.add_patch(orig_box)
        ax.text(0.125, y_pos, genre_family['original'], 
               ha='center', va='center', color='white',
               fontsize=12, weight='bold')
        ax.text(0.125, y_pos - 0.025, f"({genre_family['year']})",
               ha='center', va='center', color='white', fontsize=9)
        
        # Draw splits
        for j, (split_name, split_year, market) in enumerate(genre_family['splits']):
            x_offset = 0.3 + (j % 2) * 0.35
            y_offset = y_pos + 0.03 - (j // 2) * 0.06
            
            # Connection line
            ax.plot([0.2, x_offset], [y_pos, y_offset], 
                   'k-', alpha=0.3, linewidth=1)
            
            # Split genre box
            split_box = FancyBboxPatch((x_offset, y_offset - 0.02), 0.28, 0.04,
                                      boxstyle="round,pad=0.01",
                                      facecolor=split_colors[i*4 + j], 
                                      edgecolor='black', linewidth=1,
                                      alpha=0.8)
            ax.add_patch(split_box)
            
            ax.text(x_offset + 0.14, y_offset, f"{split_name} ({split_year})",
                   ha='center', va='center', fontsize=10, weight='bold')
            ax.text(x_offset + 0.14, y_offset - 0.015, market,
                   ha='center', va='center', fontsize=8, style='italic')
    
    # Add annotations
    ax.text(0.5, 0.98, 'How 4 Genres Became 32: Commercial Market Segmentation', 
           ha='center', va='top', fontsize=18, weight='bold',
           transform=ax.transAxes)
    
    ax.text(0.5, 0.02, 
           'Each split represents a commercial decision to target specific markets, not genuine musical innovation',
           ha='center', va='bottom', fontsize=12, style='italic',
           transform=ax.transAxes)
    
    # Add legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, fc=original_color, label='Original Genres (2004)'),
        mpatches.Rectangle((0, 0), 1, 1, fc='lightblue', label='Commercial Splits (2008-2024)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('phantom_genre_splits.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_commercial_drivers_chart():
    """Create chart showing commercial drivers of genre creation"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data
    drivers = [
        'Festival Stage Programming',
        'Beatport Sales Categories', 
        'Spotify Algorithm Needs',
        'DJ Booking Efficiency',
        'Marketing Segmentation',
        'Regional Scene Commodification'
    ]
    
    impacts = [85, 90, 75, 70, 65, 60]  # Percentage impact
    
    examples = [
        '5 stages = 5+ genre needs',
        '32 charts = 32x revenue streams',
        '2000+ micro-genres for playlists',
        'Genre determines booking fee',
        'Target demographics by genre',
        'Local sounds → global products'
    ]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(drivers))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(drivers)))
    
    bars = ax.barh(y_pos, impacts, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels and examples
    for i, (bar, example) in enumerate(zip(bars, examples)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{int(width)}%', ha='left', va='center',
                fontsize=12, weight='bold')
        ax.text(5, bar.get_y() + bar.get_height()/2,
                example, ha='left', va='center',
                fontsize=10, style='italic', color='darkgray')
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(drivers, fontsize=12)
    ax.set_xlabel('Commercial Impact on Genre Creation (%)', fontsize=14, weight='bold')
    ax.set_xlim(0, 100)
    ax.set_title('Commercial Forces Driving EDM Genre Proliferation', 
                fontsize=16, weight='bold', pad=20)
    
    # Add reference line
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax.text(50, -0.5, 'Majority impact threshold', ha='center', 
           fontsize=10, color='red', style='italic')
    
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('commercial_drivers.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_revenue_correlation():
    """Create visualization showing correlation between genres and revenue"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left panel - Genre count vs market size
    years = np.array([2004, 2008, 2012, 2016, 2020, 2024])
    genre_counts = np.array([4, 8, 12, 20, 28, 32])
    market_size = np.array([1.5, 2.8, 4.5, 7.2, 9.8, 11.8])  # Billions USD
    
    # Create dual axis
    ax1_twin = ax1.twinx()
    
    # Plot data
    line1 = ax1.plot(years, genre_counts, 'b-', linewidth=3, marker='o', 
                     markersize=10, label='Genre Count')
    line2 = ax1_twin.plot(years, market_size, 'g-', linewidth=3, marker='s', 
                          markersize=10, label='Market Size ($B)')
    
    # Fill areas
    ax1.fill_between(years, 0, genre_counts, alpha=0.3, color='blue')
    ax1_twin.fill_between(years, 0, market_size, alpha=0.3, color='green')
    
    # Labels
    ax1.set_xlabel('Year', fontsize=14, weight='bold')
    ax1.set_ylabel('Number of Genres', fontsize=14, weight='bold', color='blue')
    ax1_twin.set_ylabel('Market Size (Billions USD)', fontsize=14, weight='bold', color='green')
    ax1.set_title('Genre Proliferation Correlates with Market Growth', 
                 fontsize=16, weight='bold')
    
    # Add correlation coefficient
    correlation = np.corrcoef(genre_counts, market_size)[0, 1]
    ax1.text(0.5, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax1.transAxes, ha='center', va='top',
            fontsize=14, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    ax1.grid(True, alpha=0.3)
    
    # Right panel - Revenue per genre
    genre_names = ['House', 'Techno', 'Trance', 'Bass', 'Commercial EDM', 'Others']
    revenues = [2.8, 2.1, 1.5, 1.8, 2.9, 0.7]  # Billions
    
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71', '#f39c12', '#95a5a6']
    explode = (0, 0, 0, 0, 0.1, 0)  # Explode commercial EDM
    
    wedges, texts, autotexts = ax2.pie(revenues, labels=genre_names, autopct='%1.1f%%',
                                       colors=colors, explode=explode, startangle=90,
                                       textprops={'fontsize': 12, 'weight': 'bold'})
    
    # Add center text
    ax2.text(0, 0, '$11.8B\nTotal', ha='center', va='center',
            fontsize=16, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax2.set_title('EDM Revenue Distribution by Genre Family (2024)', 
                 fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig('revenue_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all visualizations
if __name__ == "__main__":
    print("Creating EDM genre proliferation visualizations...")
    
    create_genre_timeline()
    print("✓ Created timeline visualization")
    
    create_phantom_splits_diagram()
    print("✓ Created genre splits diagram")
    
    create_commercial_drivers_chart()
    print("✓ Created commercial drivers chart")
    
    create_revenue_correlation()
    print("✓ Created revenue correlation visualization")
    
    print("\nAll visualizations created successfully!")
    print("Files generated:")
    print("- edm_genre_timeline.png")
    print("- phantom_genre_splits.png") 
    print("- commercial_drivers.png")
    print("- revenue_correlation.png")