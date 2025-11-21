from metpy.calc import wet_bulb_temperature, relative_humidity_from_dewpoint
import numpy as np


def add_zero_crossings(profile, heights):
    
    pre_crossing_level = np.argwhere((np.diff(np.sign(profile)) != 0)*1).flatten()
    post_crossing_level = pre_crossing_level + 1
    indices = []
    crossings_m = []
    for pre, post in zip(pre_crossing_level, post_crossing_level):
        xp = profile[pre:post+1]
        fp = heights[pre:post+1]
        if xp[0] > xp[1]:
            xp = xp[::-1]
            fp = fp[::-1]
        crossings_m.append(np.interp(0, xp, fp))
        indices.append(post)
    profile = np.insert(profile, indices, 0)
    heights = np.insert(heights, indices, crossings_m)

    return profile, heights



import numpy as np


def calc_sounding_stats(profile, heights):

    FREEZING_K = 273.15
    GRAVITY = 9.81 

    cold_area, warm_area = [], []
    low_i, cold_thickness, warm_thickness = 0, 0, 0
    profile, heights = add_zero_crossings(profile, heights)
    surface = profile[0]
    try:
        upper_bound_index = np.argwhere(profile==0).max() + 1 # get index of highest crossing where we no longer care about
        lower_bound_index = np.argwhere(profile==0).min()
        highest_crossing = heights[upper_bound_index - 1]
        lowest_crossing = heights[lower_bound_index]
    except:
        min_cold, max_warm, t_span, highest_crossing, lowest_crossing = np.nan, np.nan, np.nan, np.nan, np.nan
        metrics = dict(cold_area=np.sum(cold_area),
                       warm_area=np.sum(warm_area),
                       cold_thickness=cold_thickness,
                       warm_thickness=warm_thickness,
                       min_cold=min_cold, 
                       max_warm=max_warm, 
                       highest_crossing=highest_crossing,
                       lowest_crossing=lowest_crossing, 
                       surface=surface)
        return metrics
    
    min_cold = profile[:upper_bound_index].min()
    max_warm = profile[:upper_bound_index].max()
    for i in range(upper_bound_index):
        if (profile[i] == 0):

            energy = np.trapz(GRAVITY * (profile[low_i:i+1]/FREEZING_K) , heights[low_i:i+1])
            if energy <= 0:
                cold_area.append(energy)
                cold_thickness += heights[i] - heights[low_i]
            else:
                warm_area.append(energy)
                warm_thickness += heights[i] - heights[low_i]
            low_i = i

        metrics[cold_area] = np.sum(cold_area
        metrics[warm_area] = np.sum(warm_area)
        metrics[cold_thickness] = cold_thickness
        metrics[warm_thickness] = warm_thickness
        metrics[min_cold] = min_cold
        metrics[max_warm] = max_warm
        metrics[highest_crossing] = highest_crossing
        metrics[lowest_crossing] = lowest_crossing
        metrics[surface] = surface

    return metrics



def bourgouin_ip_frzr(t_ca, t_wa):

    if t_ca <= (56 + (t_wa * 0.66)):
        return 3
    else:
        return 2


def calc_bourgouin_ptype(t_profile, h_profile):

    temp_profile, height_profile = add_zero_crossings(t_profile, h_profile)
    metrics = calc_sounding_stats(temp_profile, height_profile)
    NA, PA, CT, WT, min_C, max_C, max_cross_m, min_cross_m, sfc = metric.values()


    n_crossings = np.where(temp_profile == 0, 1, 0).sum()

    if (n_crossings == 0) and (sfc >= 0):
        return 0
        
    elif (n_crossings == 0) and (sfc < 0):
        return 1

    if (n_crossings == 1) and (sfc > 0):
        
        if PA > 13.2:
            return 0         
        else:
            return 1
            
    if (n_crossings == 2) and (sfc <= 0):
        
        if PA < 2:
            return 1
        else:   
            return bourgouin_ip_frzr(NA, PA)
            
    if n_crossings >= 3:
        
        if PA_list[-1] < 2:

            if sfc >= 0:
                return 0
            else:
                return 1
        else:
            return bourgouin_ip_frzr(NA, PA)
            

def calc_modified_bourgouin_ptype(t_profile, td_profile, twb_profile, h_profile):

    prob_ice = calc_prob_ice(t_profile, td_profile)
    wb_profile, height_profile = add_zero_crossings(twb_profile, h_profile)
    metrics = calc_sounding_stats(wb_profile, height_profile)
    NA, PA, CT, WT, min_C, max_C, max_cross_m, min_cross_m, sfc = metrics.values()
    prob_snow_i = 1540 * np.exp(-(0.29 * PA))
    prob_snow = (prob_ice / 100) * prob_snow_i
    if PA > 0:
        prob_icep_i = (((2.3 * NA) - (42 * np.log(PA + 1)) + 3) * prob_ice)
    else:
        prob_icep_i = 0
    prob_icep = (prob_ice / 100) * prob_icep_i
    prob_frzr_i = (-2.1 * NA) + (0.2 * PA) + 458
    if PA < 5:
        prob_frzr_i = prob_frzr_i * 0.2 * PA
    prob_frzr = (100 - prob_ice) + ((prob_ice / 100) * prob_frzr_i)
    
    if sfc > 0:
        prob_rain = prob_frzr
        prob_frzr = 0
    else:
        prob_rain = 0

    ptypes = softmax(np.array([prob_rain, prob_snow, prob_icep, prob_frzr]) / 100)
    cat_pred = np.argwhere(ptypes==np.max(ptypes)).flatten()[-1]

    return ptypes, cat_pred
        

def calc_prob_ice(t_profile, td_profile, height_resolution=250):

    SAT_LAYER_THRESH = 1000
    UNSAT_LAYER_THRESH = 1500

    rh_i = relative_humidity_from_dewpoint(t_profile * units.degC, td_profile * units.degC, phase='solid').magnitude[::-1]
    # print(min(rh_i), max(rh_i))
    saturated = np.where(rh_i > 0.75, 1, 0)
    unsaturated = np.where(saturated == 1, 0, 1)

    layer_top_i, layer_bottom_i = get_layer_indices(arr=saturated, min_length=np.ceil(SAT_LAYER_THRESH / height_resolution) + 1)

    if layer_top_i is not None:

        sfc_layer_top_i, sfc_layer_bottom_i = get_layer_indices(arr=unsaturated[layer_bottom_i:], min_length=np.ceil(UNSAT_LAYER_THRESH / height_resolution) + 1)
        if sfc_layer_top_i is not None:
            return 0

        min_layer_temp = min(t_profile[::-1][layer_top_i:layer_bottom_i + 1])

        if min_layer_temp <+ -15:
            
            return 100
            
        elif (min_layer_temp > -15) & (min_layer_temp < -7):
            
            return (-0.065 * min_layer_temp ** 4) - (3.1544 * min_layer_temp ** 3) - (56.414 * min_layer_temp ** 2) - (449.6 * min_layer_temp) - (1308)
    
        else:
    
            return 0
    
    else:
        return 0

def get_layer_indices(arr, min_length=5):
    
    start = None
    for i in range(len(arr)):
        if arr[i] == 1:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_length:
                return (start, i - 1)
            start = None

    # Check if array ends with a valid run
    if start is not None and len(arr) - start >= min_length:
        return (start, len(arr) - 1)

    return (None, None)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
