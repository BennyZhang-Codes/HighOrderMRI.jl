# Per-readout LowRank

Each readout retains its own local temporal and spatial factors. No shared
spatial compression or final shared recompression is performed. This is the
per-readout singular-vector representation used as a MaxGIRF-style comparison;
it does not imply reproduction of every component of the original MaxGIRF
pipeline, hardware model or solver.

```julia
op = HighOrderLowRankOp(grid, kspha, times;
    method=:per_readout,
    fieldmap, csm, mask, arrayType=Array,
    L_rank=8, rsvd_oversample=5, rsvd_seed=1234,
)
```

`L_rank` is the retained rank of each readout; `L_rank+rsvd_oversample` is the
local sketch width. It must fit both the readout sample count and masked voxel
count. Shared compression tolerances do not control this representation.

## Factor storage

| `per_readout_storage` | Storage and work |
|:--|:--|
| `:gpu` | Original default; complete factors on the selected array backend, including CPU with `arrayType=Array`. |
| `:cpu` | Full spatial bank in host RAM; bounded GPU slots stream ahead of use. |
| `:regenerate` | Retain Q/Z and regenerate spatial factors into bounded GPU slots. |

The latter two require CUDA. A CPU spatial bank requires approximately
`sizeof(Complex{T}) * count(mask) * L_rank * nDyn` bytes, before the temporal
factors and other arrays. Streaming bounds device storage but still retains
that host bank. Regeneration avoids the full bank; the default host factor
cache budget is zero and regeneration adds computation.

```julia
op = HighOrderLowRankOp(grid, kspha, times;
    method=:per_readout,
    fieldmap, csm, mask, arrayType=CuArray, gpus=[0,1],
    L_rank=10, rsvd_oversample=5,
    rsvd_backend=:kernel, rsvd_finalize=:gram, rsvd_distribution=:voxel,
    per_readout_storage=:regenerate, per_readout_cache_size=5,
    per_readout_host_cache_bytes=0,
    normal_distribution=:readout,
)
```

This CUDA configuration is illustrative. Select only authorized, available
visible device IDs. Readout scheduling and factor storage are independent
choices. `normal_distribution=:readout` assigns whole readout groups, while
`:channel` assigns coil groups. The optional batched shared-basis normal is
not a Per-readout execution method.

Hold the local rank, seed, data, physical model, weights and solver fixed when
comparing with Incremental Shared. Then compare storage/scheduling changes
separately. See [factor equations](/theory/low-rank#independent-per-readout-representation),
[parameters](/reference/highorderlowrankop#per-readout-parameters), and
[resource cleanup](/reference/resources).
