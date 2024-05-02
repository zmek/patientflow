from matplotlib import pyplot as plt

def prob_dist_plot(prob_dist_data, title_, directory_path = None, figsize = (6,3), include_titles = False, truncate_at_beds=20, text_size = None):
            
    plt.figure(figsize=figsize)
    file_name = title_.replace(' ', '_').replace('/n', '_').replace('%', 'percent') + '.png'
    plt.bar(prob_dist_data.index[0:truncate_at_beds+1], 
            prob_dist_data['agg_proba'].values[0:truncate_at_beds+1])
    
    if text_size:

        plt.tick_params(axis='both', which='major', labelsize=text_size)
    
    if include_titles:
        
        plt.title(title_, fontsize = text_size)
        plt.xlabel('Number of beds')
        plt.ylabel('Probability')
        
    plt.tight_layout()
    
    if directory_path:
        plt.savefig(directory_path / file_name, dpi=300)
    plt.show()
