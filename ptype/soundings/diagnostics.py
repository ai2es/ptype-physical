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
