import numpy as np
from scipy.spatial.distance import pdist,squareform
import pirt
import scipy.cluster.hierarchy
from tqdm import tqdm
from multiday_suite2p.utils import deform_masks, create_mask_img, add_overlap_info

def transform_points(xpix,ypix, deform):
    """Transform points (in pixel space) according to DeformationField object
    
    Arguments:
        xpix {numpy array}          -- x pixel locations (1d)
        ypix {numpy array}          -- y pixel locations (1d)
        deform {DeformationField}   -- Deformation field
    
    Returns:
        [numpy array]               -- transformed point list (y,x order; num_points x 2)
    """
    pp = np.vstack([xpix,ypix]).transpose() 
    pp = pp.astype(np.float64)
    v = deform.get_field_in_points(pp,1)
    pp[:,0]-=v
    v = deform.get_field_in_points(pp,0)
    pp[:,1]-=v  
    return pp[:,[1,0]] # y,x order.

def register_sessions(images, settings):
    """Registers session images with DiffeomorphicDemonsRegistration and returns deformation object.
    
    Arguments:
        images {list of dictionaries}        -- Overview images of sessions (size: num_sessions) 
                                        (nested dict: "mean_img", "enhanced_img", "max_img")
    
    settings {dictionary}:
        img_type {str}              -- type of image to use for registration 
                                        ("mean_img", "enhanced_img", or "max_img") (default: {"enhanced_img"})
        grid_sampling_factor {int}  -- The grid sampling of the grid at the final level. (default: {1})
        scale_sampling {int}        -- The amount of iterations for each level (default: {20})
        speed_factor {int}          -- The relative force of the transform. (default: {2})
    
    Returns:
        [list]                      -- Contains DeformationField objects (size: num_sessions) 
        [list]                      -- transformed images per plane (size: num_sessions)
                                     (nested dict: "mean_img", "enhanced_img", "max_img")
    """
    ims = [im[settings['img_type']] for im in images]
    reg = pirt.DiffeomorphicDemonsRegistration(*ims)
    reg.params.grid_sampling_factor = settings['grid_sampling_factor']
    reg.params.scale_sampling = settings['scale_sampling']
    reg.params.speed_factor = settings['speed_factor']
    reg.register(verbose=0)
    deforms = []
    # store transforms.
    trans_images = []
    for isession in range(len(images)):
        # get deform.
        deform = reg.get_deform(isession)
        deforms.append(deform)
        # transform images.
        transformed = {}
        for field in ["mean_img","enhanced_img","max_img"]:
            transformed[field] = deform.apply_deformation(
                images[isession][field])
        trans_images.append(transformed)
    return deforms, trans_images

def transform_cell_masks(deforms,masks):
    """Transforms cell masks using deformation fields from register_sessions
    
    Arguments:
    deforms {list of DeformationFields} -- Contains DeformationField objects (size: num_sessions) 
        masks {list of dictionaries}    -- Detected functional cell mask information (size: num_sessions)
                                        (dict: "xpix": array of x pixel locations, "ypix": array of y pixel locations,
                                        "neuropil_mask": array of pixel indices for neuropil mask,
                                        "lam": lambda mask weights per pixel, "med": median center of mask)    
    
    Returns:
        [numpy array]               -- Transformed functional cell mask information (size: num_sessions x num_planes)
        [list]                      -- Transformed labeled mask images per plane (size: num_planes)
    """
    im_size = deforms[0].field_shape
    trans_masks = []
    trans_label = []
    for isession, deform in tqdm(enumerate(deforms),total=len(deforms)):
        session_masks = deform_masks(masks[isession],deform)
        # add session number and set id to 0 (unassigned/not clustered)
        session_masks= [dict(item, **{'session':isession,'id':0}) for item in session_masks]
        # store session masks.
        trans_masks.append(session_masks)
        # create label image.
        trans_label.append(create_mask_img(session_masks,im_size,mark_overlap=True))
    return trans_masks, trans_label

def square_to_condensed(i, j, n):
    """Converts squareform indices to condensed form index
    (see : https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist)
    
    Arguments:
        i {int} -- index 1
        j {int} -- index 2
        n {int} -- number of entries (or length of one dimension of the squareform matrix)
    
    Returns:
        int -- index into condensed distance matrix
    """
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return int(n*j - j*(j+1)/2 + i - 1 - j)

def cluster_cell_masks(masks,im_size, settings,verbose=True):
    """Clusters cell masks across sessions using jaccard distance matrix.
    
    Arguments:
        masks {list of dictionaries}     -- All cell masks information (size: num_sessions)
                                        (dict: "xpix": array of x pixel locations, "ypix": array of y pixel locations, 
                                        "ipix": array of linear index locations, "npix": number of pixels in mask,
                                        "lam": lambda mask weights per pixel, "med": median center of mask,
                                        "session","plane,"id")     
        im_size {list}                  -- Image size combined plane
    
    Settings Keys:
        min_distance {float}        -- minimal euclidean distance between cell masks to be considered for clustering
        criterion {string}          -- criterion used for clustering (default: "distance")    
        threshold {float}           -- threshold used for clustering (default: {0.975})
        min_sessions {int}          -- exclude masks not present for this number of times (default: {2})
        step_sizes: {list}          -- clustering happens in these sizes blocks across the plane (for memory reasons)
        bin_size: {int}             -- Look for masks around center+bin-size to avoid edge cases
        min_distance: {int}         -- only masks with centers within this pixel radius of each other are considered for clustering.
    
    Returns:
        list                        -- list of putative cell masks (size: num_putative_cells).
                                        each item contains list of clustered cell masks.
        np.array                    -- image of cell masks 
    """

    putative_cells = []
    counter=int(0)
    for ypos in tqdm(range(0,im_size[0],settings['step_sizes'][1]),disable=verbose==False):
        for xpos in range(0,im_size[1],settings['step_sizes'][0]):
            # collect unassigned masks in range
            cell_info = np.array([ cell for session in masks for cell in session \
                        if (cell["id"]==0) and (cell["med"][0]>ypos-settings['bin_size']) and (cell["med"][1]>xpos-settings['bin_size']) and
                            (cell["med"][0]<ypos+settings['step_sizes'][0]+settings['bin_size']) and 
                            (cell["med"][1]<xpos+settings['step_sizes'][1]+settings['bin_size'])])     
            num_cells = len(cell_info)
            if num_cells>0:
                # get nearby masks.
                centers = np.array([cell["med"] for cell in cell_info])
                dist = np.triu(squareform(pdist(centers)<settings['min_distance']))
                is_possible_pair = np.array(np.where(dist)).transpose()
                # calculate jaccard distances for possible pairs.
                if is_possible_pair.shape[0]>0:
                    jac_shape = int(((num_cells*num_cells)/2)-(num_cells/2))
                    jac_mat = np.ones(jac_shape)*10000  
                    for pair in is_possible_pair:
                        if cell_info[pair[0]]["session"]!=cell_info[pair[1]]["session"]:
                            num_both = np.intersect1d(cell_info[pair[0]]["ipix"],cell_info[pair[1]]["ipix"],assume_unique=True).shape[0]
                            jac_mat[square_to_condensed(pair[0], pair[1], num_cells)] = 1-(num_both/(cell_info[pair[0]]["ipix"].shape[0] + cell_info[pair[1]]["ipix"].shape[0] - num_both))
                    # cluster.
                    Z = scipy.cluster.hierarchy.complete(jac_mat)
                    clust = scipy.cluster.hierarchy.fcluster(Z, t=settings['threshold'],criterion=settings['criterion'])
                    # remove cells below minimum presence.
                    uni,counts = np.unique(clust,return_counts=True)
                    min_sessions = int(np.ceil((settings['min_sessions_perc']/100)*len(masks)))
                    clust[np.isin(clust,uni[counts<min_sessions])]=0
                    # select only found clusters within bin.
                    uni = np.unique(clust)
                    for clust_id in uni:
                        if clust_id!=0:
                            idx = clust==clust_id
                            med = centers[idx].mean(axis=0)
                            if (med[0]>=ypos) and (med[0]<ypos+settings['step_sizes'][0]) and\
                                (med[1]>=xpos) and (med[1]<xpos+settings['step_sizes'][1]):
                                    counter+=1
                                    adj_cell_info = []
                                    for cell in cell_info[idx]:
                                        new_cell = dict(cell)# otherwise changes in place and cant rerun this function.
                                        new_cell["id"]=counter
                                        adj_cell_info.append(new_cell)
                                    putative_cells.append(adj_cell_info)
    # Create result label images.
    label_im = np.zeros([len(masks),im_size[0],im_size[1]],np.uint32)
    for masks in putative_cells:
        for mask in masks:
            label_im[[mask["session"] for i in mask["xpix"]],mask["ypix"],mask["xpix"]] = masks[0]["id"]     
    return putative_cells, label_im 

def create_template_masks(putative_cells,im_size,settings):
    """Create averaged template mask for each group of clustered cell masks (putative cells)
    
    Arguments:
        putative_cells {list}       -- list of putative cell masks (size: num_putative_cells).
                                        each item contains list of clustered cell masks.
        im_size {list}              -- iamge plane size (num_planes)
    
    settings keys:
        min_perc {int}              -- create template based on pixels detected in X% of sesions
    
    Returns:
        [list]                      -- list of template mask information (size: num_putative_cells).
                                        (dict: "xpix": array of x pixel locations, "ypix": array of y pixel locations, 
                                        "ipix": array of linear index locations, "npix": number of pixels in mask,
                                        "lam": lambda mask weights per pixel, "med": median center of mask,
                                        "num_sessions","plane","id")             
        [np.array]                  -- Image of template cell masks.

    """
    template_masks = []
    # get masks possibly belonging to one cell.
    for masks in putative_cells:
        # get cell mask pixels present in atleast more then <min_perc>% of sessions.
        idx = np.hstack([mask["ipix"] for mask in masks])
        lam = np.hstack([mask["lam"] for mask in masks])
        unique, counts = np.unique(idx, return_counts=True)
        filt_idx = unique[(counts / len(masks))>(settings['min_perc']/100)]
        # calculate cell mask properties.
        pixs = np.unravel_index(filt_idx, im_size)
        xpix = pixs[1] # list of values (not single value)
        ypix = pixs[0]
        med = [np.median(ypix),np.median(xpix)]
        radius = np.asarray([ mask['radius'] for mask in masks]).mean()
        avg_lem = [lam[idx==i].mean() for i in filt_idx]
        # store.
        template_masks.append({
            "id":     masks[0]["id"],
            "ipix": filt_idx,
            "xpix": xpix,
            "ypix": ypix,
            "med": med,
            "lam": np.array(avg_lem),
            "radius": radius,
            "num_sessions": len(masks)
        })
    # add overlap info.
    template_masks = add_overlap_info(template_masks)
    # filter out small masks.
    before_size = len(template_masks)
    template_masks = [mask for mask in template_masks if (len(mask['ipix'])-sum(mask['overlap']))>=settings['min_size_non_overlap']]
    print(f"Before filtering: #{before_size} cells, after: #{len(template_masks)} cells")
    template_im = create_mask_img(template_masks,im_size,mark_overlap=True)
    return template_masks, template_im

def backward_transform_masks(templates, deforms):
    """Perform backward transform of cell masks back to original sample space (unregistered).
    
    Arguments:
        templates {list}            -- list of filtered template masks (size: num_cells)
                                        (dict: "xpix": array of x pixel locations, "ypix": array of y pixel locations, 
                                        "ipix": array of linear index locations, "npix": number of pixels in mask,
                                        "lam": lambda mask weights per pixel, "num_sessions","plane","id")                
        deforms {list of DeformationFields}       -- Contains registration DeformationField objects (size: num_sessions). 
    
    Returns:
        list                        -- cell mask information (num_sessions)
                                        contains embedded list of dictionaries (size: num_cells) 
                                        (dict: "xpix": array of x pixel locations, "ypix": array of y pixel locations, 
                                        "ipix": array of linear index locations, "npix": number of pixels in mask,
                                        "lam": lambda mask weights per pixel, "overlap": bool of overlapping pixels)
        list                        -- list of images of cell ids per session (size: num_sessions).
        list                        -- list of images of lambda weights per session (size: num_sessions)                        
    """
    trans_masks = []
    deform_lam_imgs=[]
    deform_label_imgs=[]
    im_size = deforms[0][0].shape
    for deform in tqdm(deforms):
        # deform masks backwards.
        session_masks = deform_masks(templates,deform.as_backward_inverse())
        trans_masks.append(session_masks)
        # Create mask images.
        deform_label_imgs.append(create_mask_img(session_masks, im_size ,mark_overlap=True))
        deform_lam_imgs.append(create_mask_img(session_masks,im_size ,field="lam"))
    return trans_masks, deform_label_imgs, deform_lam_imgs
