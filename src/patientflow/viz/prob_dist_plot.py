from matplotlib import pyplot as plt
import numpy as np
import itertools

def prob_dist_plot(prob_dist_data, title_, directory_path = None, figsize = (6,3), include_titles = False, truncate_at_beds=20, text_size = None, bar_colour='#5B9BD5', file_name = None, min_beds_lines = None):
            
    plt.figure(figsize=figsize)
    
    if not file_name:
        file_name = title_.replace(' ', '_').replace('/n', '_').replace('%', 'percent') + '.png'
    plt.bar(prob_dist_data.index[0:truncate_at_beds+1], 
            prob_dist_data['agg_proba'].values[0:truncate_at_beds+1],
            color=bar_colour)
    
    plt.xlim(-0.5, truncate_at_beds + 0.5)
    plt.xticks(np.arange(0, truncate_at_beds + 1, 5))  # Set x-axis ticks at every 5 units

    if min_beds_lines:

        colors = itertools.cycle(plt.cm.gray(np.linspace(0.3, 0.7, len(min_beds_lines))))

        for point in min_beds_lines:
            plt.axvline(x=min_beds_lines[point], linestyle='--', linewidth=2, color=next(colors), label=f'{point*100:.0f}% probability')

        plt.legend(loc='upper right')
    
    if text_size:

        plt.tick_params(axis='both', which='major', labelsize=text_size)
    
    if include_titles:
        
        plt.title(title_, fontsize = text_size)
        plt.xlabel('Number of beds')
        plt.ylabel('Probability')
        
    plt.tight_layout()
    
    if directory_path:
        plt.savefig(directory_path / file_name.replace(' ','_'), dpi=300)
    plt.show()