// MY METAL REFERENCE — things I keep forgetting

// --- thread IDs ---
/*uint id       [[ thread_position_in_grid ]]*/        // global thread index
/*uint local_id [[ thread_position_in_threadgroup ]]*/ // index within group
/*uint group_id [[ threadgroup_position_in_grid ]]*/   // which group am I in
/*uint gsize    [[ threads_per_threadgroup ]]*/         // group size

// --- memory address spaces ---
//device    // GPU global memory   — slow, large, persistent
//threadgroup // shared within group — fast, small, temporary
/*constant */ // read-only global    — cached, good for small params

// --- the barrier — always needed after threadgroup writes ---
//threadgroup_barrier(mem_flags::mem_threadgroup);

// --- simdgroup op — your future weapon ---
//simdgroup_matrix_multiply_accumulate(C, A, B, C);

// --- dispatch from CPU side ---
// one thread per element
//[enc dispatchThreads:grid threadsPerThreadgroup:group];
// one threadgroup per element (for reductions)
//[enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
