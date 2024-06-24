import logging
import os
from matplotlib import colors as mcolors

from ptype.plotting import get_tle_files, load_data, plot_hrrr_ptype, plot_ptype, plot_probability, plot_uncertainty
from ptype.plotting import plot_winds, plot_temp, plot_dpt, plot_sp, save


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Example usage:
    base_path = '/glade/campaign/cisl/aiml/ptype_historical/winter_2023_2024/hrrr'
    valid_time = '2023-12-16 0700'
    time = valid_time.replace(' ', '_')
    n_members = 18
    output_dir = f'/glade/work/sreiner/ptype_plots/{time}/'

    # check if path exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    variables = [
        'u10', 'v10', 'ML_rain', 'ML_crain', 'ML_snow', 'ML_csnow', 'ML_frzr', 'ML_cfrzr', 'ML_icep', 'ML_cicep',
        'crain', 'csnow', 'cfrzr', 'cicep', 'ML_rain_ale', 'ML_rain_epi', 'ML_snow_ale', 'ML_snow_epi', 'ML_frzr_ale',
        'ML_frzr_epi', 'ML_icep_ale', 'ML_icep_epi', 'd2m', 't2m', 'sp'
    ]
    title = f''
    outname = f''
    
    if n_members > 1:
        title += f'Time Lagged Ensemble'
        outname += f'time_lagged_'
    
    ptypes = ['rain', 'snow', 'frzr', 'icep']

    # custom purple colormap:
    custom_colors=['#f8f4f8', '#f005fc']
    purples = mcolors.LinearSegmentedColormap.from_list("custom", custom_colors)

    cmaps = ['Greens', 'Blues', 'Reds', purples]

    files = get_tle_files(base_path, valid_time, n_members)
    ds = load_data(files, variables)
    logger.info(f'{len(files)} files loaded')
    
    title_ptype = f'{title} {valid_time} Precip\nSnow = Blue, Rain = Green, Sleet = Purple, Freezing Rain = Red'
    out_ptype = f'{output_dir}{outname}ptype_{time}'
    
    plot_hrrr_ptype(ds, cmaps, ptypes)
    save(title_ptype, f'{out_ptype}_hrrr.png')

    plot_ptype(ds, ptype=None, cmap=cmaps, ptypes=ptypes)
    save(title_ptype, f'{out_ptype}.png')
    
    plot_winds(ds, plot_ptype, cmap=cmaps, ptypes=ptypes)
    save(title_ptype, f'{out_ptype}_barbs.png')
    
    plot_temp(ds, plot_ptype, cmap=cmaps, ptypes=ptypes)
    save(title_ptype, f'{out_ptype}_temp.png')

    plot_dpt(ds, plot_ptype, cmap=cmaps, ptypes=ptypes)
    save(title_ptype, f'{out_ptype}_dpt.png')

    plot_sp(ds, plot_ptype, cmap=cmaps, ptypes=ptypes)
    save(f'{title_ptype}', f'{out_ptype}_sp.png')
    
    for ptype in ptypes:
        
        logger.info(f'plotting prob {ptype}')
        title_prob = f'{title} probability {ptype} {valid_time}'
        out_prob = f'{output_dir}{outname}prob_{time}_{ptype}'
        plot_probability(ds, ptype)
        save(title_prob, f'{out_prob}.png')
        
        plot_winds(ds, plot_probability, ptype=ptype)
        save(f'{title_prob}', f'{out_prob}_barbs.png')

        plot_temp(ds, plot_probability, ptype=ptype)
        save(f'{title_prob}', f'{out_prob}_temp.png')

        plot_dpt(ds, plot_probability, ptype=ptype)
        save(f'{title_prob}', f'{out_prob}_dpt.png')
        
        logger.info(f'plotting uncertainty {ptype}')
        title_uncert = [f'{title} Aleatoric Uncertainty {ptype} {valid_time}', f'{title} Epistemic Uncertainty {ptype} {valid_time}']
        out_uncert = f'{output_dir}{outname}uncert_{time}_{ptype}'
        ax = plot_uncertainty(ds, ptype)
        save(title_uncert, f'{out_uncert}.png', ax)

        ax = plot_winds(ds, plot_uncertainty, ptype=ptype)
        save(title_uncert, f'{out_uncert}_barbs.png', ax)
        
        ax = plot_temp(ds, plot_uncertainty, ptype=ptype)
        save(title_uncert, f'{out_uncert}_temp.png', ax)

        ax = plot_dpt(ds, plot_uncertainty, ptype=ptype)
        save(title_uncert, f'{out_uncert}_dpt.png', ax)

    