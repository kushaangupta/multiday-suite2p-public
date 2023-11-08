import numpy as np
from suite2p.extraction import preprocess

def demix_traces(F, Fneu, cell_masks, ops):
    """Demix activity from overlaping cells

    Args:
        F (numpy array): Raw fluoresence activity (size: num cells x num frames)
        Fneu ([type]): Raw neuropil activity
        cell_masks (list of dictionaries): description of cell masks (size: num cells)
                must contains keys "xpix", "ypix", "lam", and "overlap"
        ops (dictionary): Parameters for demixing (must contain "baseline",'"win_baseline", "sig_baseline", and "fs")
        l2_reg (float, optional): L2 regularization factor. Defaults to 0.01.

    Returns:
        [type]: [description]
    """
    # subtract neuropil signal and subtract baseline.
    Fcorr = F - ops['neucoeff']*Fneu
    Fbase = preprocess(Fcorr, ops['baseline'], ops['win_baseline'],
                       ops['sig_baseline'], ops['fs']) # baseline subtracted signal.
    #Collect mask information.
    num_cells = len(cell_masks) 
    Ly, Lx = ops['Ly'], ops['Lx']
    lammap = np.zeros((num_cells, Ly, Lx), np.float32) # weight mask for each mask
    Umap = np.zeros((num_cells, Ly, Lx), bool) # binarized weight masks
    covU = np.zeros((num_cells,num_cells), np.float32) # holds covariance matrix.
    for ni,mask in enumerate(cell_masks):
        ypix, xpix, lam = mask['ypix'], mask['xpix'], mask['lam']
        norm = lam.sum()
        Fbase[ni] *= norm
        lammap[ni,ypix,xpix] = lam
        Umap[ni,ypix,xpix] = True
        covU[ni,ni] = (lam**2).sum()
    #Create covariance matrix of the masks.
    for ni,mask in enumerate(cell_masks):
        if mask['overlap'].sum() > 0:
            ioverlap = mask['overlap']
            yp, xp, lam = mask['ypix'][ioverlap], mask['xpix'][ioverlap], mask['lam'][ioverlap]
            njs, ijs = np.nonzero(Umap[:, yp, xp])
            for nj in np.unique(njs):
                if nj!=ni:
                    inds = ijs[njs==nj]
                    covU[ni, nj] = (lammap[nj, yp[inds], xp[inds]] * lam[inds]).sum() #  each entry i,j is the sum of (weights in mask_i * weights in mask_j that overlap). this is an overlap score matrix in a sense
    #Solve for demixed traces of the cells. 
    #the equation we're solving is the movie M is a multiplication of the masks with the fluorescence V: M = U @ V.T . 
    #We have  U @ M.T = Fbase , and the solution for V with linear regression is np.linalg.solve(U @ U.T, U @ M.T) 
    #so we plug in for U @ M.T with Fbase and get the final equation
    l2 = np.diag(covU).mean() * ops['l2_reg']
    Fdemixed = np.linalg.solve(covU + l2*np.eye(num_cells), Fbase) 

    return Fdemixed, Fbase, covU, lammap