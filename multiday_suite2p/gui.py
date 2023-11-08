import matplotlib.pyplot as plt
from ipywidgets import HBox,VBox
import ipywidgets as widgets
from random import random
import numpy as np
from IPython.display import display

def show_imgs_with_masks(sessions,images,mask_sets, aspect_ratio=1.5):
    """GUI for looking at session images and overlay different sets of cell masks.

    Args:
        sessions (list of dictionaries): list of dictionaries with session info('date', 'session_id'). Length is number of sessions
        images (list of dictionaries): list of dictionaries. Each entry had different image type ('mean_img','enhanced_img','max_img')
        mask_sets (dictionary): dictionary containing different images corresponding to cell masks (pixel value is cell mask identity)
        aspect_ratio (float, optional): aspect ratio to show images in. Defaults to 1.5.
    """
    mask_set_names = list(mask_sets.keys())
    
    if isinstance(images[0], dict): 
        multiple_img_types=True
        img_type_names = list(images[0].keys())
    else: 
        multiple_img_types=False
        img_type_names = ['']
    # Setup UI.
    session_ui = widgets.IntSlider(min=0,max=len(sessions)-1,step=1,value=0, continuous_update=True, description='Session:')
    img_ui = widgets.Dropdown( options=img_type_names,value=img_type_names[0], description='Img Type:')
    set_ui = widgets.Dropdown( options=mask_set_names,value=mask_set_names[0], description='Mask Type:')
    opacity_ui = widgets.FloatSlider(min=0,max=1,step=0.1,value=0.5, continuous_update=True, description='Mask Opacity:')
    masks_ui = widgets.Checkbox(True,description='Show Cell Masks')
    if multiple_img_types: ui= HBox([ VBox([img_ui,session_ui]),masks_ui,VBox([set_ui,opacity_ui])])
    else: ui= HBox([ VBox([session_ui]),masks_ui,VBox([set_ui,opacity_ui])])

    # colormap.
    vals = np.linspace(0,1,10000)
    np.random.seed(4)
    np.random.shuffle(vals)
    cmap = plt.cm.colors.ListedColormap(plt.cm.hsv(vals))

    #Setup figure
    fig = plt.figure(figsize=(6,6),dpi=150)
    ax = fig.subplots()
    ax.axis('off')
    if multiple_img_types:
        handle_main = ax.imshow(images[0]["mean_img"], cmap='gray',interpolation='none')
    else:
        handle_main = ax.imshow(images[0], cmap='gray',interpolation='none')
    label_mask = mask_sets[mask_set_names[0]][0]
    label_mask = np.ma.masked_where(label_mask==0, label_mask)
    handle_overlay = ax.imshow(label_mask, cmap=cmap, alpha=0.5,interpolation='none',vmin=1,vmax=20000)
    ax.set_aspect(aspect_ratio)
    plt.tight_layout()
    fig.canvas.header_visible = False
    #fig.canvas.footer_visible = False
    # Interactive function.
    def f(session,img_type,mask_set,show_masks,opacity):

        # set title.
        ax.set_title(f"date: {sessions[session]['date']}, session: {sessions[session]['session_id']}", fontsize=12)
        # show iamge with overlay.
        if multiple_img_types: handle_main.set_data(images[session][img_type])
        else: handle_main.set_data(images[session])
        if show_masks:
            label_mask = mask_sets[mask_set]
            if isinstance(label_mask, list): label_mask=label_mask[session]
            if isinstance(label_mask, np.ndarray): 
                if label_mask.ndim==3:
                    label_mask=label_mask[session] 
            label_mask = np.ma.masked_where(label_mask==0, label_mask)
        else:
            label_mask = np.ma.masked_where(np.zeros((1,1))==0, np.zeros((1,1)))
        handle_overlay.set_data(label_mask)
        handle_overlay.set_alpha(opacity)
        handle_main.autoscale()
        pass

    out = widgets.interactive_output(f, {'session': session_ui,'img_type':img_ui,'mask_set':set_ui,'show_masks':masks_ui,'opacity':opacity_ui})
    display(ui, out)