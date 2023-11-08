# multiday-suite2p
Code for multi-day registration of suite2p data

# Usage
notebooks/multiday_registration.ipynb contains the full workflow to take data processed by suite2p and align them across sessions.
Found cell rois in each session will be clustered and filtered based on morphology, spatial overlap and detection rate across registered sessions. Clustered rois are then converted into one 'consensus' template roi and traces are recalculate from the raw binary data.

Note: Code is heavily optimized to work in parallel on the Janelia LSF cluster.

## Settings
Settings are stored in associated yml files. See an example in the 'settings' folder and below.

### alignment settings
```yml
# Server settings.
server:
  host: 'login2.int.janelia.org'
  n_cores: 10
  username: 'user'

# Suite2p cell detection filtering.
cell_detection:
  prob_threshold: 0.85 # detected cells need to have a classifier score above this threshold
  max_size: 1000 # cells with roi above this many pixels are excluded.
  stripe_borders: [462, 924] # x location of stripe borders(for compatibility)
  stripe_margin: 30 # cells too close to stripe edge are excluded (can cause registration issues)

# multiday registration through port settings
registration:
  img_type: "enhanced_img" #type of image to use for registration
  grid_sampling_factor: 1 #The grid sampling at the final level.
  scale_sampling: 20 #The amount of iterations for each level
  speed_factor: 3 #The relative force of the transform 

# cell mask clustering.
clustering:
  criterion: "distance" # criterion used for clustering
  threshold: 0.75 # Threshold used for clustering algorithm
  min_sessions_perc: 50 #Exclude masks not present for this percentage of sessions (0-100).
  min_perc: 50 #Create template based on pixels detected in X% of sesions.
  step_sizes: [200,200] # clustering happens in these sizes blocks across the plane (for memory reasons)
  bin_size: 50 # Look for masks around center+bin-size to avoid edge cases
  min_distance: 20 # only masks with centers within this pixel radius of each other are considered for clustering.
  min_size_non_overlap: 25 # minimum size of template mask in pixels.

# demixing settings.
demix:
  baseline: 'maximin' # baselining method (valid: 'maximin','constant', 'constant_prctile').
  win_baseline: 60.0 # window (in seconds) for max filter.
  sig_baseline: 10 # width of Gaussian filter in seconds.
  l2_reg: 0.1 # l2 regularization.
  neucoeff: 0.7 # for neuropil subtraction.

```

### File info
```yml
# Data folder locations.
data:
  # Contents of data root should be organized as 2020_01_01/1
  local_bin_root: '//nrs/spruston/Monitor-Test/raw/Test-Tyche'   # Local Raw Data root folder with bin files.
  server_bin_root: ''  # Server Raw Data root folder with bin files. (linux). Leave empty for same as local folder (eg: '')
  local_processed_root: '//nrs/spruston/Monitor-Test/processed/Test-Tyche'  # Processed Data folder on local machine
  server_processed_root:  ''  # Processed Data folder on server (linux). Leave empty for same as local folder (eg: '')
  suite2p_folder: suite2p
  output_folder: multi_day_demix # output folder for multiday data.
  # filter for specific session (leave '[]' for all)
  # valid examples:
  # [['2020_01_01']]
  # [['2020_01_01/0']]
  # [['2020_01_01/0', '2020_01_10']]
  # or combinations of these:
  # [['2020_01_01/0', '2020_01_10'], ['2020_01_13/1']]
  # all selections are INCLUSIVE (including edges) so ['2020_01_01','2020_01_03'] processes 3 days.
  session_selection: [['2021_10_22','2021_10_27']] 
  # Include session as standalone data for vr2p processing.
  # These sessions are not included in the registration process but theyre own data (with their own unique cell masks will be included)
  # example: ['2021_12_24/2'] leave empty:[] for none
  individual_sessions: []

# Animal Info.
animal:
  date_of_birth: "01-01-2020"
  gender: male

```

## Install
1. Install module inside suite2p conda environment.
```console
pip install -e .
``` 
2. For trace extraction make sure the file bash-scripts/extract_session_job.sh is reachable on cluster root.