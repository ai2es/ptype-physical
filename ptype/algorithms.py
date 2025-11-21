def bourgouin_ip_frzr(t_ca, t_wa):

    if t_ca <= (56 + (t_wa * 0.66)):
        return 3
    else:
        return 2


def calc_bourgouin_ptype(t_profile, h_profile):

    temp_profile, height_profile = add_zero_crossings(t_profile, h_profile)
    NA_list, PA_list, CT, WT, min_C, max_C, max_cross_m, min_cross_m, sfc = calc_sounding_stats(temp_profile, height_profile)
    NA = np.abs(np.sum(NA_list))
    PA = np.sum(PA_list)


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
    NA_list, PA_list, CT, WT, min_C, max_C, max_cross_m, min_cross_m, sfc = calc_sounding_stats(wb_profile, height_profile)
    NA = np.abs(np.sum(NA_list))
    PA = np.sum(PA_list)
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
    
    ### NEEEDS ATTENTION
    if sfc > 0:
        prob_rain = prob_frzr
        prob_frzr = 0
    else:
        prob_rain = 0

    # ptypes = np.clip(np.array([prob_rain, prob_snow, prob_icep, prob_frzr]), a_min=0, a_max=100) / 100
    # ptypes = ptypes / np.sum(ptypes)

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
