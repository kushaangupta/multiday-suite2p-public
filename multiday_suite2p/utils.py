import numpy as np
import os
import glob
import json
from ScanImageTiffReader import ScanImageTiffReader
import yaml
from suite2p.run_s2p import default_ops
import pirt
from skimage.measure import regionprops, find_contours
import scipy.ndimage

def create_mask_img(masks,im_size,field=None,mark_overlap=False,contours=False,contour_upsampling = 1):
    """Function for creating (label) images from cells masks info.

    Args:
        masks (list of dicitonaries): Dictionaries must contain 'xpix' 'ypix'
        im_size (list): size of image to create.
        field ([string], optional): Instead of mask id use this field. Defaults to None.
        mark_overlap (bool, optional): show overlaping regions as fixed value (100). Defaults to False. Ignored if contours is True.
        contours (bool,optional): show contours of masks (disables mark_overlap)
        contour_upsampling (int,optional): upscale contours image by this factor (helpful for generating pretty overlay images)

    Returns:
        [numpy array]: resulting image.
    """
    # create holder image.
    if (not field) or  (field=="id"):
        im = np.zeros([im_size[0]*contour_upsampling,im_size[1]*contour_upsampling],np.uint32)
    else:
        im = np.zeros([im_size[0],im_size[1]], np.float64)
    for id, mask in enumerate(masks):
        # get value to fill in
        if field:
            value = mask[field]
        else:
            value = id
        # Fill out value
        if not contours:
            im[mask["ypix"],mask["xpix"]] = value
            # mark overlap as fixed value.
            if mark_overlap:
                im[mask["ypix"][mask['overlap']],mask["xpix"][mask['overlap']]] = 100
        #contours
        else:
            # create small image with this mask
            origin = [min(mask['ypix']-1),min(mask['xpix']-1)]
            ypix, xpix = mask['ypix']-origin[0], mask['xpix']-origin[1]
            temp_img = np.zeros([max(ypix)+2,max(xpix)+2],bool)
            temp_img[ypix,xpix]=True
            temp_img = scipy.ndimage.zoom(temp_img, contour_upsampling, order=0)
            # find contours.
            contours_ind = np.vstack(find_contours(temp_img)).astype(int)
            im[contours_ind[:,0]+(origin[0]*contour_upsampling),contours_ind[:,1]+(origin[1]*contour_upsampling)] = value
    return im

def tif_metadata(image_path):
    image = ScanImageTiffReader(image_path)
    metadata_raw = image.metadata()
    metadata_str = metadata_raw.split('\n\n')[0]
    metadata_json = metadata_raw.split('\n\n')[1]
    metadata_dict = dict(item.split('=') for item in metadata_str.split('\n') if 'SI.' in item)
    metadata = {k.strip().replace('SI.','') : v.strip() for k, v in metadata_dict.items()}
    for k in list(metadata.keys()):
        if '.' in k:
            ks = k.split('.')
            # TODO just recursively create dict from .-containing values
            if k.count('.') == 1:
                if not ks[0] in metadata.keys():
                    metadata[ks[0]] = {}
                metadata[ks[0]][ks[1]] = metadata[k]
            elif k.count('.') == 2:
                if not ks[0] in metadata.keys():
                    metadata[ks[0]] = {}
                    metadata[ks[0]][ks[1]] = {}
                elif not ks[1] in metadata[ks[0]].keys():
                    metadata[ks[0]][ks[1]] = {}
                metadata[ks[0]][ks[1]] = metadata[k]
            elif k.count('.') > 2:
                print('skipped metadata key ' + k + ' to minimize recursion in dict')
            metadata.pop(k)
    metadata['json'] = json.loads(metadata_json)
    metadata['image_shape'] = image.shape()
    return metadata

def metadata_to_ops(metadata):
    data = {}
    #frame rate.
    data['fs'] = float(metadata['hRoiManager']['scanVolumeRate'])
    #number of planes.
    z_collection = metadata['hFastZ']['userZs']
    if isinstance(z_collection,str):
        data['nplanes'] = 1
    else:
        data['nplanes'] = len(z_collection)
    #number of rois.
    roi_metadata = metadata['json']['RoiGroups']['imagingRoiGroup']['rois']
    data['nrois'] = len(roi_metadata)
    #channels (NOT SURE IF THIS CORRECT BECAUSE I DONT HAVE A SESSION WITH MORE THAN ONE CHANNEL...)
    data['nchannels'] = int(metadata['hChannels']['channelsActive']) #or channelSave?
    #roi info.
    roi = {}
    w_px = []
    h_px = []
    cXY = []
    szXY = []
    for r in range(data['nrois']):
        roi[r] = {}
        roi[r]['w_px'] = roi_metadata[r]['scanfields']['pixelResolutionXY'][0]
        w_px.append(roi[r]['w_px'])
        roi[r]['h_px'] = roi_metadata[r]['scanfields']['pixelResolutionXY'][1]
        h_px.append(roi[r]['h_px'])
        roi[r]['center'] = roi_metadata[r]['scanfields']['centerXY']
        cXY.append(roi[r]['center'])
        roi[r]['size'] = roi_metadata[r]['scanfields']['sizeXY']
        szXY.append(roi[r]['size'])
    w_px = np.asarray(w_px)
    h_px = np.asarray(h_px)
    szXY = np.asarray(szXY)
    cXY = np.asarray(cXY)
    cXY = cXY - szXY / 2
    cXY = cXY - np.amin(cXY, axis=0)
    mu = np.median(np.transpose(np.asarray([w_px, h_px])) / szXY, axis=0)
    imin = cXY * mu
    imin = np.ceil(imin)
    #
    n_rows_sum = np.sum(h_px)
    n_flyback = (metadata['image_shape'][1] - n_rows_sum) / np.max([1, data['nrois'] - 1])
    irow = np.insert(np.cumsum(np.transpose(h_px) + n_flyback), 0, 0)
    irow = np.delete(irow, -1)
    irow = np.vstack((irow, irow + np.transpose(h_px)))
    #lines.
    data['dx'] = []
    data['dy'] = []
    data['lines'] = []
    for i in range(data['nrois']):
        data['dy'] = np.hstack((data['dy'], imin[i,1]))
        data['dx'] = np.hstack((data['dx'], imin[i,0]))
        data['lines'].append(np.array(range(irow[0,i].astype('int32'), irow[1,i].astype('int32'))))
    data['lines'] = np.array(data['lines'])
    data['dx'] = data['dx'].astype('int32')
    data['dy'] = data['dy'].astype('int32')
    return data

def yaml_to_dict(file_path):
    with open(file_path) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

def multiday_ops(exp,session,folder_name,settings):
    print(session)
    data_path = os.path.join(exp['data']['folder_linux'],session['date'],str(session['sub_dir']))
    # read meta data from tiff.
    tif_path = glob.glob(os.path.join(data_path,f"*{session['date']}_{session['sub_dir']}_*.tif"))
    if len(tif_path)==0:
        raise NameError(f"Could not find tif in {data_path}")
    ops = metadata_to_ops(tif_metadata(tif_path[0]))
    # path locations.
    ops['data_path'] = [data_path]
    ops['save_path0'] = data_path
    ops['look_one_level_down'] = False
    ops['save_folder'] = folder_name
    ops['fast_disk'] = os.path.join(ops['save_path0'],ops['save_folder']) # stores binary here.
    ops = {**settings,**ops}
    # add optional suite2p settings.
    ops = {**default_ops(),**ops}
    return ops

def create_cropped_deform_field(deform, ori, crop_size):
    """Crops out part of a Pirt DeformationField

    Args:
        deform ([pirt.DeformationObject]): Deformation field.
        ori (np array): origin of crop region (min x, mix y) size:2x1
        crop_size (list): full size of cropped field.

    Returns:
        DeformationObject: Crop deformation field.
        np.array: origin of deformation field.
    """
    crop_size = np.array(crop_size)
    im_size = deform[0].shape
    # make sure origin isnt negative.
    ori[ori<0]=0
    # make sure crop doenst extend over size of deformation field.
    for dim in range(2):
        if ori[dim]+crop_size[dim]>im_size[dim]:
            ori[dim]=im_size[dim]-crop_size[dim] 
    # create deformation field from crop.
    crop_deform = pirt.DeformationFieldBackward([deform[0][ori[0]:ori[0]+crop_size[0] ,ori[1]:ori[1]+crop_size[1]],
                deform[1][ori[0]:ori[0]+crop_size[0] ,ori[1]:ori[1]+crop_size[1]]])
    return crop_deform,ori

def deform_masks(masks,deform,crop_bin = 500):
    """Deforms cell masks according to Pirt DeformationField

    Args:
        masks (list): list of dictionaries must contain keys 'xpix','ypix', and 'lam'
        deform (DeformationField): Deformation to apply.

    Returns:
        [list of ditionaries]: Deformed masks containing keys 'xpix','ypix','lam','ipix','radius', and 'med'
    """
    deformed_masks = []
    for mask in masks:
        # create cropped deformation field.
        crop_size = [crop_bin,crop_bin]
        crop_deform, ori = create_cropped_deform_field(deform,np.array(mask["med"],int) - int(crop_bin/2),crop_size)
        # transform lambda weight values.
        im = np.zeros(crop_size,float)
        im[mask["ypix"]-ori[0], mask["xpix"]-ori[1]]=mask["lam"]
        imAr = pirt.Aarray(im,origin=tuple(ori))
        imAr = np.array(crop_deform.apply_deformation(imAr,interpolation=0))
        # find non zero pixels.
        pixs = np.argwhere(imAr!=0)
        lam_r = imAr[pixs[:,0],pixs[:,1]] # lambda values.
        pixs+=ori
        pixs=pixs.astype(int)
        ipix_r = np.ravel_multi_index(np.transpose(pixs),deform[0].shape).astype(int)
        # get median.
        med_r = [np.median(pixs[:,0]),np.median(pixs[:,1])]
        # get radius.
        props = regionprops((imAr>0).astype(np.uint8))
        radius_r = min([prop.minor_axis_length for prop in props])    
        # store values.
        info = {'xpix':pixs[:,1],'ypix': pixs[:,0],'ipix':ipix_r, 
                'med':med_r,'lam': lam_r,'radius': radius_r}
        deformed_masks.append(info)
    # add overlap info.
    deformed_masks = add_overlap_info(deformed_masks)
    return deformed_masks

def add_overlap_info(masks):
    """ Adds 'overlap' field to list of cell masks. 
    This marks indices in 'xpix' 'ypix' and 'ipix' that overlap with other masks.

    Args:
        masks (list): list of dictionaries. Must have field 'ipix'

    Returns:
        [list]: list of dicitonaries with added 'overlap' key.
    """
    # add overlap info.
    ipixs = np.concatenate([ mask["ipix"] for mask in masks]).astype(int) # list of pixels in all masks
    unique_pixels, count = np.unique(ipixs, return_counts=True)
    for mask in masks:
        # look for overlap.
        inds = np.searchsorted(unique_pixels,mask["ipix"])
        mask['overlap'] = count[inds]>1
    return masks
