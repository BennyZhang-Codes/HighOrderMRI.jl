module PCS2DCS

"""
    DIRECTIONS

Standard anatomical directions in the Patient Coordinate System (PCS).
"""
const DIRECTIONS = ["dSag", "dCor", "dTra"]

"""
    TRANSFORMATIONS

Transformation matrices from PCS (Patient Coordinate System) to DCS (Device Coordinate System) 
for different MRI scanner manufacturers based on the patient orientation string.
"""
const TRANSFORMATIONS = Dict(
    # Siemens device physical definition (X=Right, Y=Up, Z=Out)
    # according to IDEA manual
    "SIEMENS" => Dict(
        "HFS"  => [ 1  0  0;  0 -1  0;  0  0 -1],
        "HFP"  => [-1  0  0;  0  1  0;  0  0 -1],
        "HFDR" => [ 0  1  0;  1  0  0;  0  0 -1],
        "HFDL" => [ 0 -1  0; -1  0  0;  0  0 -1],
        "FFS"  => [-1  0  0;  0 -1  0;  0  0  1],
        "FFP"  => [ 1  0  0;  0  1  0;  0  0  1],
        "FFDR" => [ 0 -1  0;  1  0  0;  0  0  1],
        "FFDL" => [ 0  1  0; -1  0  0;  0  0  1]
    ),
    
    # Placeholder for GE device physical definitions
    "GE" => Dict(
        "HFS"  => [ 1  0  0;  0 -1  0;  0  0 -1], # Placeholder, to be replaced with actual GE matrix
        # ...
    ),
    
    # Placeholder for Philips device physical definitions
    "PHILIPS" => Dict(
        "HFS"  => [ 1  0  0;  0 -1  0;  0  0 -1], # Placeholder, to be replaced with actual Philips matrix
        # ...
    ),
    
    # Placeholder for UIH (United Imaging Healthcare)
    "UIH" => Dict(
        "HFS"  => [ 1  0  0;  0 -1  0;  0  0 -1], # Placeholder, to be replaced with actual UIH matrix
        # ...
    )
)

export DIRECTIONS, TRANSFORMATIONS

end