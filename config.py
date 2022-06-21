

class Config(object):

    FILES_FOLDER = "3D_models"

    # Global noise parameters
    noises = [.001, .002, .004, .006, .008, .01, .015, .02]

    # Decimation parameters
    decimation = [.80, .60, .40, .20]

    # Subdivision iterations
    subdivision_iterations = [1, 2, 3]

    # Holes creation parameters
    pctge_holes = [.001, .0025, .005, .0075, .01]
    DIST = [0, .01, .02, .03]

    # Noise creation parameters
    noises_dist = [.01, .02, .03, .04, .05]
    pctge_patches = [.001, .0025, .005, .0075, .01]
