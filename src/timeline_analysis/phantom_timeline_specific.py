import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_phantom_timeline():
    """Create timeline specifically showing when phantom genres split"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define phantom splits with exact years
    phantom_splits = [
        {
            'base': 'Deep House',
            'phantom': 'Organic House',
            'base_year': 2004,
            'split_year': 2020,
            'reason': 'Festival "conscious" market',
            'similarity': '75%',
            'color': '#e74c3c'
        },
        {
            'base': 'Tech House',
            'phantom': 'Minimal - Deep Tech',
            'base_year': 2008,
            'split_year': 2019,
            'reason': 'Underground purist market',
            'similarity': '79%',
            'color': '#3498db'
        },
        {
            'base': 'Techno',
            'phantom': 'Techno (Peak Time) & (Raw/Hypnotic)',
            'base_year': 2004,
            'split_year': 2022,
            'reason': 'Mood-based playlist optimization',
            'similarity': '61%',
            'color': '#2ecc71'
        },
        {
            'base': 'Trance',
            'phantom': 'Trance (Main Floor) & (Raw/Hypnotic)',
            'base_year': 2004,
            'split_year': 2022,
            'reason': 'Venue-specific booking',
            'similarity': '73%',
            'color': '#9b59b6'
        }
    ]
    
    # Key commercial events
    events = [
        (2004, 'Beatport Launch', 'bottom'),
        (2011, 'EDC Vegas Era', 'top'),
        (2015, 'Spotify EDM Explosion', 'bottom'),
        (2018, 'Festival Stage Segmentation', 'top'),
        (2020, 'Pandemic Streaming Surge', 'bottom'),
        (2024, '32 Genres / $11.8B Market', 'top')
    ]
    
    # Set up timeline
    ax.set_xlim(2002, 2026)
    ax.set_ylim(-0.5, len(phantom_splits) + 0.5)
    
    # Draw timeline base
    for i in range(len(phantom_splits)):
        ax.axhline(y=i, color='lightgray', linewidth=1, alpha=0.5)
    
    # Draw phantom splits
    for i, split in enumerate(phantom_splits):
        y_pos = i
        
        # Original genre box
        orig_box = FancyBboxPatch((split['base_year']-0.5, y_pos-0.2), 
                                  2, 0.4,
                                  boxstyle="round,pad=0.02",
                                  facecolor=split['color'], 
                                  edgecolor='black', 
                                  linewidth=2,
                                  alpha=0.8)
        ax.add_patch(orig_box)
        ax.text(split['base_year']+0.5, y_pos, split['base'],
               ha='center', va='center', fontsize=11, weight='bold')
        
        # Draw split line
        ax.plot([split['base_year']+1.5, split['split_year']-0.5], 
                [y_pos, y_pos], 
                color=split['color'], linewidth=3, alpha=0.5)
        
        # Phantom genre box
        phantom_width = 5 if '&' in split['phantom'] else 3
        phantom_box = FancyBboxPatch((split['split_year']-0.5, y_pos-0.2), 
                                    phantom_width, 0.4,
                                    boxstyle="round,pad=0.02",
                                    facecolor=split['color'], 
                                    edgecolor='black', 
                                    linewidth=2,
                                    alpha=0.4,
                                    linestyle='--')
        ax.add_patch(phantom_box)
        ax.text(split['split_year']+phantom_width/2-0.5, y_pos, split['phantom'],
               ha='center', va='center', fontsize=10, style='italic')
        
        # Add similarity badge
        similarity_x = split['split_year'] + phantom_width + 0.2
        ax.text(similarity_x, y_pos, split['similarity'] + ' similar',
               ha='left', va='center', fontsize=10, weight='bold',
               color='darkred',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                        edgecolor='darkred', linewidth=1))
        
        # Add reason
        ax.text(2025, y_pos, split['reason'], ha='right', va='center',
               fontsize=9, style='italic', color='gray')
    
    # Add commercial events
    for year, event, position in events:
        y = len(phantom_splits) if position == 'top' else -0.5
        
        # Event marker
        ax.scatter(year, y, s=200, color='gold', edgecolor='black', 
                  linewidth=2, marker='*', zorder=10)
        
        # Event label
        va = 'bottom' if position == 'top' else 'top'
        y_offset = 0.1 if position == 'top' else -0.1
        ax.text(year, y + y_offset, event, ha='center', va=va,
               fontsize=10, weight='bold', rotation=20,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='gold', linewidth=2))
    
    # Labels and title
    ax.set_xlabel('Year', fontsize=14, weight='bold')
    ax.set_yticks(range(len(phantom_splits)))
    ax.set_yticklabels([f"Case {i+1}" for i in range(len(phantom_splits))])
    ax.set_title('The Timeline of Phantom Genre Creation: Commercial Splits, Not Musical Evolution', 
                fontsize=18, weight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.8, label='Original Genre'),
        mpatches.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.4, 
                          linestyle='--', label='Phantom Split'),
        mpatches.Circle((0, 0), 0.5, fc='gold', label='Commercial Event')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    # Add key insight
    ax.text(0.5, -0.08, 
           'Each split coincides with commercial needs: festival programming, streaming algorithms, or market segmentation.\n' +
           'The high acoustic similarity (61-79%) proves these are marketing categories, not genuine musical differences.',
           transform=ax.transAxes, ha='center', va='top',
           fontsize=12, style='italic',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('phantom_timeline_specific.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_beatport_evolution():
    """Show Beatport's genre explosion over time"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Data for Beatport genre additions
    beatport_timeline = [
        (2004, ['House', 'Techno', 'Trance', 'Drum & Bass'], 'Launch'),
        (2008, ['Minimal', 'Progressive House', 'Electro House'], 'First expansion'),
        (2012, ['Dubstep', 'Trap', 'Deep House'], 'EDM boom'),
        (2016, ['Big Room', 'Future House', 'Bass House'], 'Festival market'),
        (2018, ['140/Deep Dubstep/Grime', 'Bass/Club', 'UK Bass'], 'Bass fragmentation'),
        (2020, ['Organic House/Downtempo', 'Afro House', 'Melodic House'], 'Niche markets'),
        (2022, ['Techno (Peak Time)', 'Techno (Raw)', 'Trance (Raw)'], 'Mood categories'),
        (2024, ['Amapiano', 'Hard Techno', 'Indie Dance'], 'Regional commodification')
    ]
    
    # Create visualization
    y_pos = 0
    colors = plt.cm.viridis(np.linspace(0, 1, len(beatport_timeline)))
    
    for i, (year, genres, phase) in enumerate(beatport_timeline):
        # Year marker
        ax.text(0, y_pos, str(year), ha='right', va='center',
               fontsize=14, weight='bold')
        
        # Phase label
        phase_box = FancyBboxPatch((0.5, y_pos-0.15), 3, 0.3,
                                  boxstyle="round,pad=0.02",
                                  facecolor=colors[i], 
                                  edgecolor='black',
                                  alpha=0.3)
        ax.add_patch(phase_box)
        ax.text(2, y_pos, phase.upper(), ha='center', va='center',
               fontsize=11, weight='bold')
        
        # Genre additions
        for j, genre in enumerate(genres):
            x = 4 + (j % 3) * 3.5
            y = y_pos + (j // 3) * 0.3 - 0.15
            
            genre_box = FancyBboxPatch((x, y-0.1), 3.2, 0.2,
                                      boxstyle="round,pad=0.02",
                                      facecolor=colors[i], 
                                      edgecolor='black',
                                      alpha=0.7)
            ax.add_patch(genre_box)
            ax.text(x+1.6, y, genre, ha='center', va='center',
                   fontsize=10)
        
        y_pos -= 1
    
    # Add cumulative count
    cumulative = [4, 7, 10, 13, 16, 19, 22, 25]
    for i, (count, y) in enumerate(zip(cumulative, range(0, -len(beatport_timeline), -1))):
        ax.text(14.5, y, f'Total: {count}', ha='center', va='center',
               fontsize=12, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='black', linewidth=2))
    
    # Title and labels
    ax.set_xlim(-1, 16)
    ax.set_ylim(y_pos-0.5, 0.5)
    ax.set_title("Beatport's Genre Inflation: How 4 Became 32 Through Commercial Pressure", 
                fontsize=18, weight='bold')
    ax.text(0.5, 0.95, 
           'Each expansion targeted specific market segments, not musical movements',
           transform=ax.transAxes, ha='center', fontsize=12, style='italic')
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('beatport_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate visualizations
if __name__ == "__main__":
    create_phantom_timeline()
    print("✓ Created phantom genre timeline")
    
    create_beatport_evolution()
    print("✓ Created Beatport evolution chart")