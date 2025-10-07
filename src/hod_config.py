# hod_config.py

# Dictionaries to resolve column names in HDF5/TXT files.
CANONICAL_ORDER = ["x", "y", "z", "vx", "vy", "vz", "logM", "conc", "id"]

# Typical synonyms (HDF5 dtype.names or TXT headers)
H5_NAME_MAP_WITH_CONC = {
    "x":   ["x", "X", "pos_x", "position_x"],
    "y":   ["y", "Y", "pos_y", "position_y"],
    "z":   ["z", "Z", "pos_z", "position_z"],
    "vx":  ["vx", "VX", "vel_x", "velocity_x"],
    "vy":  ["vy", "VY", "vel_y", "velocity_y"],
    "vz":  ["vz", "VZ", "vel_z", "velocity_z"],
    "logM":["logM", "log_mass", "log10_mass", "Mlog10", "M_log10"],
    "conc":["conc", "concentration", "c", "cNFW"],
    "id":  ["id", "ID", "halo_id", "haloid", "HaloID"]
}

# Same as above but without concentration
H5_NAME_MAP_NO_CONC = {
    k: v for k, v in H5_NAME_MAP_WITH_CONC.items() if k != "conc"
}

# For TXT: if there is a header with names, we reuse the same synonyms
TXT_NAME_MAP_WITH_CONC = H5_NAME_MAP_WITH_CONC
TXT_NAME_MAP_NO_CONC   = H5_NAME_MAP_NO_CONC

# For TXT without header, assume standard order by positions:
# WITH concentrations (9 cols)
TXT_POS_WITH_CONC = {
    "x":0, "y":1, "z":2, "vx":3, "vy":4, "vz":5, "logM":6, "conc":7, "id":8
}
# WITHOUT concentrations (8 cols)
TXT_POS_NO_CONC = {
    "x":0, "y":1, "z":2, "vx":3, "vy":4, "vz":5, "logM":6, "id":7
}
