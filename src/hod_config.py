# hod_config.py

# Dictionaries to resolve column names in HDF5/TXT files.
CANONICAL_ORDER = ["x", "y", "z", "vx", "vy", "vz", "logM", "Rvir", "Rs", "id"]

# Typical synonyms (HDF5 dtype.names or TXT headers)
H5_NAME_MAP_WITH_R = {
    "x":   ["x", "X", "pos_x", "position_x"],
    "y":   ["y", "Y", "pos_y", "position_y"],
    "z":   ["z", "Z", "pos_z", "position_z"],
    "vx":  ["vx", "VX", "vel_x", "velocity_x"],
    "vy":  ["vy", "VY", "vel_y", "velocity_y"],
    "vz":  ["vz", "VZ", "vel_z", "velocity_z"],
    "logM":["logM", "log_mass", "log10_mass", "Mlog10", "M_log10"],
    "Rvir":["Rvir", "R_vir", "virial_radius"],
    "Rs":  ["Rs", "R_s", "scale_radius"],
    "id":  ["id", "ID", "halo_id", "haloid", "HaloID"]
}

# Same as above but without concentration
H5_NAME_MAP_NO_R = {
    k: v for k, v in H5_NAME_MAP_WITH_R.items() if k != "Rvir" and k != "Rs"
}

# For TXT: if there is a header with names, we reuse the same synonyms
TXT_NAME_MAP_WITH_R = H5_NAME_MAP_WITH_R
TXT_NAME_MAP_NO_R   = H5_NAME_MAP_NO_R

# For TXT without header, assume standard order by positions:
# WITH radius (10 cols)
TXT_POS_WITH_R = {
    "x":0, "y":1, "z":2, "vx":3, "vy":4, "vz":5, "logM":6, "Rvir":7, "Rs":8, "id":9
}
# WITHOUT radius (8 cols)
TXT_POS_NO_R = {
    "x":0, "y":1, "z":2, "vx":3, "vy":4, "vz":5, "logM":6, "id":7
}
