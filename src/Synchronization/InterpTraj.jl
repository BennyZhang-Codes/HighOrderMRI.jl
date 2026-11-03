"""
    InterpTrajTime(traj::AbstractArray{T, 2}, dt::Real, delTime::Real, datatime::AbstractVector{T}; intermode::Interpolations.InterpolationType = AkimaMonotonicInterpolation()) where T <: Real

Interpolate field coefficients from equally spaced field samples to requested
acquisition times. `traj` is `(nFieldSample,nTerm)` and `datatime` is an ADC
time vector of the same element type `T`; the result is a host matrix
`(length(datatime),nTerm)` converted to `T`.

Sample j is assigned time `(j-1)*dt-delTime`. Therefore a positive `delTime`
evaluates the unshifted field at `datatime+delTime`. Use one consistent time
unit; seconds are appropriate when these times multiply a field map in Hz.
Outside the shifted measured interval, the result is zero, not extrapolated
field dynamics. The default is Akima monotonic interpolation applied to each
column. Use a floating-point `T`, a positive interval and enough field samples
for the selected interpolation method. This function does not integrate the
coefficients or convert their physical units.
"""
function InterpTrajTime(
    traj      ::AbstractArray{T, 2}      , 
    dt        ::Real                     , 
    delTime   ::Real                     ,
    datatime  ::AbstractVector{T}        ;
    intermode ::Interpolations.InterpolationType = AkimaMonotonicInterpolation()
    ) where {T<:Real}
    nSample, nTerm = size(traj)

    # Calculate adjusted time (delayed)
    trajTim = dt * (0:nSample-1) .- delTime

    # Perform Akima interpolation (assuming 1D interpolation for each row of the trajectory)
    traj_interp = []
    for i in 1:nTerm  # Loop over each column (dimension) of the trajectory
        itp = interpolate(trajTim, traj[:, i], intermode)
        etp = extrapolate(itp, 0)  # Extrapolate to the beginning of the time vector
        push!(traj_interp, etp.(datatime))  # Store the interpolated result
    end

    # Convert the result back to an array (if necessary)
    return T.(hcat(traj_interp...))  # Combine the interpolated columns into a single matrix
end

"""
    InterpTrajTime(traj::AbstractArray{T, 2}, dt::Real, delTime::Real; intermode::Interpolations.InterpolationType = AkimaMonotonicInterpolation()) where T <: Real

Convenience overload that requests the original equally spaced times
`dt*(0:nFieldSample-1)`. It retains the zero-extrapolation and positive-delay
convention of the explicit-time overload.

The generated time vector inherits the arithmetic type of `dt`. Currently a
Float32 trajectory with Float64 `dt` can fail dispatch to the explicit-time
overload, which requires matching trajectory/time element types. Use `dt` of
the trajectory's floating-point type, or pass an explicit same-type ADC vector.
The return shape is the input trajectory shape.
"""
function InterpTrajTime(
    traj      ::AbstractArray{T, 2}      , 
    dt        ::Real                     , 
    delTime   ::Real                     ;
    intermode ::Interpolations.InterpolationType = AkimaMonotonicInterpolation()
    ) where {T<:Real}
    nSample, nTerm = size(traj)
    # Create time vector for input trajectory
    datatime = dt * (0:nSample-1)
    return InterpTrajTime(traj, dt, delTime, datatime; intermode=intermode)
end


"""
    InterpTrajTime(traj::AbstractArray{T, 3}, dt::Real, delTime::AbstractVector{T}, datatime::AbstractArray{T, 2}; intermode::Interpolations.InterpolationType = AkimaMonotonicInterpolation()) where T <: Real

Interpolate each shot independently. `traj` is `(nShot,nFieldSample,nTerm)`,
`delTime` has one offset per shot, and `datatime` is `(nShot,nADCSample)`.
Offsets and ADC times must share the trajectory element type `T`; the result
is a host array `(nShot,nADCSample,nTerm)` of that type. Each shot uses the
two-dimensional helper, including its sign convention and zero extrapolation.
Different field and ADC sample counts are permitted by this interpolation API.
This is independent of the more restrictive current multishot delay estimator.
"""
function InterpTrajTime(
    traj      ::AbstractArray{T, 3}      , 
    dt        ::Real                     , 
    delTime   ::AbstractVector{T}        ,
    datatime  ::AbstractArray{T, 2}      ;
    intermode ::Interpolations.InterpolationType = AkimaMonotonicInterpolation()
    ) where {T<:Real}
    @assert size(traj, 1) == length(delTime) "Number of shots in trajectory must match length of delTime vector"
    @assert size(datatime, 1) == length(delTime) "Number of shots in datatime must match length of delTime vector"

    nShot, _, nTerm = size(traj)
    traj_interp = []
    for i in 1:nShot
        shot_interp = InterpTrajTime(traj[i, :, :], dt, delTime[i], datatime[i, :]; intermode=intermode)
        push!(traj_interp, shot_interp)
    end
    traj_out = Array{T}(undef, nShot, size(datatime, 2), nTerm)
    for ishot in 1:nShot
        traj_out[ishot,:,:] = traj_interp[ishot]
    end
    return traj_out
end