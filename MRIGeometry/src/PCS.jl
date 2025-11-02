module PCS

"""
    DIRECTIONS

Standard anatomical directions in the Patient Coordinate System (PCS).
"""
const DIRECTIONS = ["dSag", "dCor", "dTra"]

"""
    TRANSFORMATIONS

Transformation matrices from PCS (Patient Coordinate System) to DCS (Device Coordinate System)
based on the patient orientation string.
Reference: IDEA manual and DICOM standard
"""
const TRANSFORMATIONS = Dict(
    "HFS"  => [ 1  0  0;  0 -1  0;  0  0 -1],
    "HFP"  => [-1  0  0;  0  1  0;  0  0 -1],
    "HFDR" => [ 0  1  0;  1  0  0;  0  0 -1],
    "HFDL" => [ 0 -1  0; -1  0  0;  0  0 -1],
    "FFS"  => [-1  0  0;  0 -1  0;  0  0  1],
    "FFP"  => [ 1  0  0;  0 -1  0;  0  0 -1],
    "FFDR" => [ 0 -1  0;  1  0  0;  0  0  1],
    "FFDL" => [ 0  1  0; -1  0  0;  0  0  1],
)

export DIRECTIONS, TRANSFORMATIONS

end