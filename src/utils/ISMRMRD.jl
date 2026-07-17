# module ISMRMRD

using HDF5
import HDF5: hdf5_type_id
using LinearAlgebra

export Dataset, Acquisition, Image, EncodingCounters, AcquisitionHeader, ImageHeader
export read_xml_header, write_xml_header
export append_acquisition, read_acquisition, append_image, read_image
export IMTYPE_MAGNITUDE, IMTYPE_PHASE, IMTYPE_REAL, IMTYPE_COMPLEX

# ====================================================================
# 1. Official ISMRMRD Constants (Matches ismrmrd.h)
# ====================================================================
const IMTYPE_MAGNITUDE = UInt16(1)
const IMTYPE_PHASE     = UInt16(2)
const IMTYPE_REAL      = UInt16(3)
const IMTYPE_COMPLEX   = UInt16(4)

const EXP_DATA_FLOAT   = UInt16(5)
const EXP_DATA_CXFLOAT = UInt16(6)

# ====================================================================
# 2. Low-level fixed-size C-Structs 
# (Strictly matches C++ memory layout for HDF5 Compound types)
# ====================================================================

# 2.1 Encoding Counters
struct EncodingCounters
    kspace_encode_step_1 :: UInt16
    kspace_encode_step_2 :: UInt16
    average              :: UInt16
    slice                :: UInt16
    contrast             :: UInt16
    phase                :: UInt16
    repetition           :: UInt16
    set                  :: UInt16
    segment              :: UInt16
    user                 :: NTuple{8, UInt16}
end
EncodingCounters() = EncodingCounters(0,0,0,0,0,0,0,0,0, ntuple(i->UInt16(0), 8))

# 2.2 Acquisition Header (340 bytes)
struct AcquisitionHeader
    version                :: UInt16
    flags                  :: UInt64
    measurement_uid        :: UInt32
    scan_counter           :: UInt32
    acquisition_time_stamp :: UInt32
    physiology_time_stamp  :: NTuple{3, UInt32}
    number_of_samples      :: UInt16
    available_channels     :: UInt16
    active_channels        :: UInt16
    channel_mask           :: NTuple{16, UInt64}
    discard_pre            :: UInt16
    discard_post           :: UInt16
    center_sample          :: UInt16
    encoding_space_ref     :: UInt16
    trajectory_dimensions  :: UInt16
    sample_time_us         :: Float32
    position               :: NTuple{3, Float32}
    read_dir               :: NTuple{3, Float32}
    phase_dir              :: NTuple{3, Float32}
    slice_dir              :: NTuple{3, Float32}
    patient_table_position :: NTuple{3, Float32}
    idx                    :: EncodingCounters
    user_int               :: NTuple{8, Int32}
    user_float             :: NTuple{8, Float32}
end

function AcquisitionHeader(; samples=0, channels=1)
    AcquisitionHeader(
        1, 0, 0, 0, 0, ntuple(i->UInt32(0), 3),
        UInt16(samples), UInt16(channels), UInt16(channels),
        ntuple(i->UInt64(0), 16), 0, 0, UInt16(div(samples, 2)), 0, 0, 2.5f0,
        (0.0f0, 0.0f0, 0.0f0), (1.0f0, 0.0f0, 0.0f0), (0.0f0, 1.0f0, 0.0f0), (0.0f0, 0.0f0, 1.0f0),
        (0.0f0, 0.0f0, 0.0f0), EncodingCounters(),
        ntuple(i->Int32(0), 8), ntuple(i->0.0f0, 8)
    )
end

# 2.3 Image Header (198 bytes)
struct ImageHeader
    version                :: UInt16
    data_type              :: UInt16
    flags                  :: UInt64
    measurement_uid        :: UInt32
    matrix_size            :: NTuple{3, UInt16}
    field_of_view          :: NTuple{3, Float32}
    channels               :: UInt16
    position               :: NTuple{3, Float32}
    read_dir               :: NTuple{3, Float32}
    phase_dir              :: NTuple{3, Float32}
    slice_dir              :: NTuple{3, Float32}
    patient_table_position :: NTuple{3, Float32}
    average                :: UInt16
    slice                  :: UInt16
    contrast               :: UInt16
    phase                  :: UInt16
    repetition             :: UInt16
    set                    :: UInt16
    acquisition_time_stamp :: UInt32
    physiology_time_stamp  :: NTuple{3, UInt32}
    image_type             :: UInt16
    image_index            :: UInt16
    image_series_index     :: UInt16
    user_int               :: NTuple{8, Int32}
    user_float             :: NTuple{8, Float32}
    attribute_string_len   :: UInt32
end

function ImageHeader(matrix_size::NTuple{3, Integer}; channels=1, imtype=IMTYPE_MAGNITUDE)
    ImageHeader(
        1, EXP_DATA_FLOAT, 0, 0,
        UInt16.(matrix_size), (0.0f0, 0.0f0, 0.0f0), UInt16(channels),
        (0.0f0, 0.0f0, 0.0f0), (1.0f0, 0.0f0, 0.0f0), (0.0f0, 1.0f0, 0.0f0), (0.0f0, 0.0f0, 1.0f0),
        (0.0f0, 0.0f0, 0.0f0), 0, 0, 0, 0, 0, 0, 0, ntuple(i->UInt32(0), 3),
        UInt16(imtype), 0, 0,
        ntuple(i->Int32(0), 8), ntuple(i->0.0f0, 8), 0
    )
end

# ====================================================================
# 3. Data Wrapper Objects (Matches MATLAB ismrmrd.Acquisition/Image)
# ====================================================================

mutable struct Acquisition
    head :: AcquisitionHeader
    data :: Matrix{ComplexF32}  # [Samples, Channels]
    traj :: Matrix{Float32}     # [TrajectoryDims, Samples]
end

mutable struct Image
    head       :: ImageHeader
    data       :: Array{Float32, 4}   # [X, Y, Z, Channels]
    attributes :: String
end

# ====================================================================
# 4. Dataset Class Management (Matches MATLAB ismrmrd.Dataset)
# ====================================================================
mutable struct Dataset
    file     :: HDF5.File
    filename :: String
    mode     :: String

    function Dataset(filename::String, mode::String="r")
        if mode == "w"
            f = h5open(filename, "w")
            create_group(f, "dataset")
        elseif mode == "r+"
            f = h5open(filename, "r+")
        else
            f = h5open(filename, "r")
        end
        new(f, filename, mode)
    end
end

# ====================================================================
# XML Header Read/Write (Robust Cross-Language Version)
# ====================================================================
function read_xml_header(dset::Dataset)
    raw_xml = read(dset.file["dataset/xml"])
    
    # Handle different HDF5 string representations from C++/Python/Julia
    if raw_xml isa Vector{String}
        return raw_xml[1]         # Official C++ ISMRMRD behavior (1D array of strings)
    elseif raw_xml isa Vector{UInt8}
        return String(raw_xml)    # Byte array behavior
    elseif raw_xml isa String
        return raw_xml            # Scalar string behavior
    else
        return string(raw_xml)    # Fallback
    end
end

function write_xml_header(dset::Dataset, xml_str::String)
    # Ensure backwards compatibility by overwriting if it exists
    if haskey(dset.file, "dataset/xml")
        delete_object(dset.file, "dataset/xml")
    end
    
    # CRITICAL: Wrap the string in a Vector [xml_str]
    # This forces HDF5.jl to write it as a 1D array of strings of size 1.
    # This perfectly mimics the official C++ ISMRMRD library's memory layout,
    # ensuring your Julia-generated MRD files can be safely read by Python/Siemens ICE.
    dset.file["dataset/xml"] = [xml_str]
end

# ====================================================================
# 5. Image Read/Write (Fully replicates MATLAB Image Group operations)
# ====================================================================

function append_image(dset::Dataset, group_name::String, img::Image)
    g_base = "dataset/$(group_name)"
    g = haskey(dset.file, g_base) ? dset.file[g_base] : create_group(dset.file, g_base)
    
    # =========================================================================
    # 🌟 CRITICAL FIX: Explicitly and strictly type the array. 
    # This completely prevents Julia from accidentally promoting to Vector{Any}
    # =========================================================================
    local final_head::Vector{ImageHeader} = Vector{ImageHeader}()
    
    if haskey(g, "data")
        # ==========================================
        # Append mode: Image group already exists
        # ==========================================
        existing_data = read(g["data"])
        existing_head_raw = read(g["header"])
        
        # Safely extract and push strictly typed elements one by one
        for h in existing_head_raw
            push!(final_head, ImageHeader(
                UInt16(h.version), UInt16(h.data_type), UInt64(h.flags), UInt32(h.measurement_uid),
                (UInt16(h.matrix_size[1]), UInt16(h.matrix_size[2]), UInt16(h.matrix_size[3])),
                (Float32(h.field_of_view[1]), Float32(h.field_of_view[2]), Float32(h.field_of_view[3])),
                UInt16(h.channels),
                (Float32(h.position[1]), Float32(h.position[2]), Float32(h.position[3])),
                (Float32(h.read_dir[1]), Float32(h.read_dir[2]), Float32(h.read_dir[3])),
                (Float32(h.phase_dir[1]), Float32(h.phase_dir[2]), Float32(h.phase_dir[3])),
                (Float32(h.slice_dir[1]), Float32(h.slice_dir[2]), Float32(h.slice_dir[3])),
                (Float32(h.patient_table_position[1]), Float32(h.patient_table_position[2]), Float32(h.patient_table_position[3])),
                UInt16(h.average), UInt16(h.slice), UInt16(h.contrast), UInt16(h.phase), UInt16(h.repetition), UInt16(h.set),
                UInt32(h.acquisition_time_stamp),
                (UInt32(h.physiology_time_stamp[1]), UInt32(h.physiology_time_stamp[2]), UInt32(h.physiology_time_stamp[3])),
                UInt16(h.image_type), UInt16(h.image_index), UInt16(h.image_series_index),
                (Int32(h.user_int[1]), Int32(h.user_int[2]), Int32(h.user_int[3]), Int32(h.user_int[4]), Int32(h.user_int[5]), Int32(h.user_int[6]), Int32(h.user_int[7]), Int32(h.user_int[8])),
                (Float32(h.user_float[1]), Float32(h.user_float[2]), Float32(h.user_float[3]), Float32(h.user_float[4]), Float32(h.user_float[5]), Float32(h.user_float[6]), Float32(h.user_float[7]), Float32(h.user_float[8])),
                UInt32(h.attribute_string_len)
            ))
        end
        
        # Prepare the newly appended header
        attr_len = UInt32(length(Vector{UInt8}(img.attributes)))
        updated_head = ImageHeader(
            img.head.version, img.head.data_type, img.head.flags, img.head.measurement_uid,
            img.head.matrix_size, img.head.field_of_view, img.head.channels,
            img.head.position, img.head.read_dir, img.head.phase_dir, img.head.slice_dir,
            img.head.patient_table_position, img.head.average, img.head.slice, img.head.contrast,
            img.head.phase, img.head.repetition, img.head.set, img.head.acquisition_time_stamp,
            img.head.physiology_time_stamp, img.head.image_type, img.head.image_index,
            img.head.image_series_index, img.head.user_int, img.head.user_float, attr_len
        )
        
        # Safely push the new element
        push!(final_head, updated_head)
        
        # Concatenate 5D image data
        new_data = cat(existing_data, img.data, dims=5)
        
        # Safe HDF5 rewrite
        delete_object(g, "data")
        delete_object(g, "header")
        
        g["data"] = new_data
        g["header"] = final_head
        
    else
        # ==========================================
        # First-time writing mode
        # ==========================================
        attr_len = UInt32(length(Vector{UInt8}(img.attributes)))
        updated_head = ImageHeader(
            img.head.version, img.head.data_type, img.head.flags, img.head.measurement_uid,
            img.head.matrix_size, img.head.field_of_view, img.head.channels,
            img.head.position, img.head.read_dir, img.head.phase_dir, img.head.slice_dir,
            img.head.patient_table_position, img.head.average, img.head.slice, img.head.contrast,
            img.head.phase, img.head.repetition, img.head.set, img.head.acquisition_time_stamp,
            img.head.physiology_time_stamp, img.head.image_type, img.head.image_index,
            img.head.image_series_index, img.head.user_int, img.head.user_float, attr_len
        )
        
        push!(final_head, updated_head)
        
        g["data"] = reshape(img.data, (size(img.data)..., 1)) 
        g["header"] = final_head 
    end
    
    # Write attributes if available
    if !isempty(img.attributes)
        # In strict HDF5, you cannot overwrite an existing dataset with '='.
        # You must delete the old node first before writing the new one.
        if haskey(g, "attributes")
            delete_object(g, "attributes")
        end
        
        g["attributes"] = img.attributes
    end
end

function read_image(dset::Dataset, group_name::String)
    g_path = "dataset/$(group_name)"
    !haskey(dset.file, g_path) && error("Image group $(group_name) does not exist!")
    
    g = dset.file[g_path]
    data = read(g["data"])
    headers = read(g["header"])
    attrs = haskey(g, "attributes") ? read(g["attributes"]) : ""
    
    images = Image[]
    num_images = size(data, 5)
    for i in 1:num_images
        # Extract single 4D image block [X, Y, Z, Channels]
        sub_data = data[:, :, :, :, i]
        push!(images, Image(headers[i], sub_data, attrs))
    end
    return images
end

function Base.close(dset::Dataset)
    close(dset.file)
end

# end # module ISMRMRD