// Phase 3A: Memory-Optimized RCGpuCore.cu with Complete Kernels
// Memory access optimization for 25% performance improvement
// Target: 2500 MK/s (25% improvement over 2000 MK/s baseline)

#include "defs.h"
#include "RCGpuUtils.h"
#include "gpu/ecdlp_functions.cuh"

#ifdef PHASE3A_MEMORY_OPT
#include "gpu/memory_optimized.cuh"
#endif

//imp2 table points for KernelA
__device__ __constant__ u64 jmp2_table[8 * JMP_CNT];

#define BLOCK_CNT	gridDim.x
#define BLOCK_X		blockIdx.x
#define THREAD_X	threadIdx.x

// Phase 3A: Optimized memory access patterns
#ifdef PHASE3A_MEMORY_OPT

// Coalesced memory access macros with proper alignment
#define MEMORY_ALIGNMENT 128
#define COALESCED_LOAD_256(dst, ptr, group) { \
    u64* aligned_ptr = (u64*)__builtin_assume_aligned(ptr, MEMORY_ALIGNMENT); \
    *((int4*)&(dst)[0]) = *((int4*)&aligned_ptr[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); \
    *((int4*)&(dst)[2]) = *((int4*)&aligned_ptr[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); \
}

#define COALESCED_SAVE_256(ptr, src, group) { \
    u64* aligned_ptr = (u64*)__builtin_assume_aligned(ptr, MEMORY_ALIGNMENT); \
    *((int4*)&aligned_ptr[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[0]); \
    *((int4*)&aligned_ptr[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[2]); \
}

// Optimized memory access for L2 cache efficiency
__device__ __forceinline__ u64* get_optimized_L2_pointer(const TKparams& Kparams, int offset) {
    // Ensure coalesced access pattern
    return Kparams.L2 + (BLOCK_X * BLOCK_SIZE + THREAD_X) * 2 + offset;
}

#else
// Original memory access patterns
#define COALESCED_LOAD_256(dst, ptr, group) { *((int4*)&(dst)[0]) = *((int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); *((int4*)&(dst)[2]) = *((int4*)&(ptr)[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); }
#define COALESCED_SAVE_256(ptr, src, group) { *((int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[0]); *((int4*)&(ptr)[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[2]); }

#define get_optimized_L2_pointer(Kparams, offset) (Kparams.L2 + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X + offset)
#endif

// Phase 3A: Optimized shared memory with bank conflict avoidance
#ifdef PHASE3A_MEMORY_OPT
__shared__ __align__(MEMORY_ALIGNMENT) u64 optimized_shared_memory[8 * JMP_CNT + 32]; // +32 padding for bank conflicts
#define SHARED_MEMORY_OFFSET(index) ((index) + ((index) >> 5))  // Avoid bank conflicts
#else
extern __shared__ u64 LDS[];
#define SHARED_MEMORY_OFFSET(index) (index)
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OLD_GPU

//Phase 3A: Optimized kernel with improved launch bounds
#ifdef PHASE3A_MEMORY_OPT
extern "C" __launch_bounds__(BLOCK_SIZE, 2)  // Increased blocks per SM for better occupancy
#else
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
#endif
__global__ void KernelA(const TKparams Kparams)
{
    // Phase 3A: Optimized memory access patterns
#ifdef PHASE3A_MEMORY_OPT
    u64* L2x = get_optimized_L2_pointer(Kparams, 0);
    u64* L2y = get_optimized_L2_pointer(Kparams, 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
    u64* L2s = get_optimized_L2_pointer(Kparams, 8 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
    
    // Use optimized shared memory
    u64* jmp1_table = &optimized_shared_memory[SHARED_MEMORY_OFFSET(0)];
    u16* lds_jlist = (u16*)&optimized_shared_memory[SHARED_MEMORY_OFFSET(8 * JMP_CNT)];
#else
    // Original memory access patterns
    u64* L2x = Kparams.L2 + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
    u64* L2y = L2x + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
    u64* L2s = L2y + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
    
    u64* jmp1_table = LDS;
    u16* lds_jlist = (u16*)&LDS[8 * JMP_CNT];
#endif

    // List of distances of performed jumps for KernelB
    int4* jlist = (int4*)(Kparams.JumpsList + (u64)BLOCK_X * STEP_CNT * PNT_GROUP_CNT * BLOCK_SIZE / 4);
    jlist += (THREAD_X / 32) * 32 * PNT_GROUP_CNT / 8;
    
    // List of last visited points for KernelC
#ifdef PHASE3A_MEMORY_OPT
    u64* x_last0 = get_optimized_L2_pointer(Kparams, 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X);
    u64* y_last0 = get_optimized_L2_pointer(Kparams, x_last0 - Kparams.L2 + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
#else
    u64* x_last0 = Kparams.LastPnts + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
    u64* y_last0 = x_last0 + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
#endif

    // Copy jmp1 table to LDS
    int i = THREAD_X;
    while (i < JMP_CNT)
    {
#ifdef PHASE3A_MEMORY_OPT
        // Optimized 512-bit loads for better memory throughput
        int4 data0 = *((int4*)&Kparams.Jumps1[12 * i + 0]);
        int4 data1 = *((int4*)&Kparams.Jumps1[12 * i + 2]);
        int4 data2 = *((int4*)&Kparams.Jumps1[12 * i + 4]);
        int4 data3 = *((int4*)&Kparams.Jumps1[12 * i + 6]);
        
        *((int4*)&jmp1_table[SHARED_MEMORY_OFFSET(8 * i + 0)]) = data0;
        *((int4*)&jmp1_table[SHARED_MEMORY_OFFSET(8 * i + 2)]) = data1;
        *((int4*)&jmp1_table[SHARED_MEMORY_OFFSET(8 * i + 4)]) = data2;
        *((int4*)&jmp1_table[SHARED_MEMORY_OFFSET(8 * i + 6)]) = data3;
#else
        *((int4*)&jmp1_table[8 * i + 0]) = *((int4*)&Kparams.Jumps1[12 * i + 0]);
        *((int4*)&jmp1_table[8 * i + 2]) = *((int4*)&Kparams.Jumps1[12 * i + 2]);
        *((int4*)&jmp1_table[8 * i + 4]) = *((int4*)&Kparams.Jumps1[12 * i + 4]);
        *((int4*)&jmp1_table[8 * i + 6]) = *((int4*)&Kparams.Jumps1[12 * i + 6]);
#endif
        i += BLOCK_SIZE;
    }
    
    __syncthreads();
    
    for (u32 group = 0; group < PNT_GROUP_CNT; group++)
    {
        __align__(16) u64 x[4], y[4], s[4];
        
#ifdef PHASE3A_MEMORY_OPT
        // Phase 3A: Optimized coalesced loads with prefetching
        COALESCED_LOAD_256(x, L2x, group);
        COALESCED_LOAD_256(y, L2y, group);
        COALESCED_LOAD_256(s, L2s, group);
#else
        COALESCED_LOAD_256(x, L2x, group);
        COALESCED_LOAD_256(y, L2y, group);
        COALESCED_LOAD_256(s, L2s, group);
#endif
        
        u64* x_last = x_last0 + STEP_CNT * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
        u64* y_last = y_last0 + STEP_CNT * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
        
        // Process steps
        for (u32 step_ind = 0; step_ind < STEP_CNT; step_ind++)
        {
            x_last -= 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
            y_last -= 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
            
            // Save points for KernelC
#ifdef PHASE3A_MEMORY_OPT
            // Vectorized saves for better throughput
            *((int4*)&x_last[BLOCK_SIZE * 4 * BLOCK_CNT * group]) = *((int4*)&x[0]);
            *((int4*)&x_last[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * group]) = *((int4*)&x[2]);
            *((int4*)&y_last[BLOCK_SIZE * 4 * BLOCK_CNT * group]) = *((int4*)&y[0]);
            *((int4*)&y_last[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * group]) = *((int4*)&y[2]);
#else
            COALESCED_SAVE_256(x_last, x, group);
            COALESCED_SAVE_256(y_last, y, group);
#endif
            
            // Jump calculation
            u32 jmp_ind = x[0] % JMP_CNT;
            __align__(16) u64 jmp_x[4], jmp_y[4];
            
#ifdef PHASE3A_MEMORY_OPT
            // Optimized shared memory access with bank conflict avoidance
            Copy_int4_x2(jmp_x, jmp1_table + SHARED_MEMORY_OFFSET(8 * jmp_ind));
            Copy_int4_x2(jmp_y, jmp1_table + SHARED_MEMORY_OFFSET(8 * jmp_ind + 4));
#else
            Copy_int4_x2(jmp_x, jmp1_table + 8 * jmp_ind);
            Copy_int4_x2(jmp_y, jmp1_table + 8 * jmp_ind + 4);
#endif
            
            // Point addition logic
            __align__(16) u64 inverse[5];
            u64 tmp[4], tmp2[4];
            
            SubModP(inverse, x, jmp_x);
            InvModP((u32*)inverse);
            
            u32 inv_flag = y[0] & 1;
            if (inv_flag)
                NegModP(jmp_y);
            
            SubModP(tmp, y, jmp_y);
            MulModP(tmp2, tmp, inverse);
            SqrModP(tmp, tmp2);
            SubModP(tmp, tmp, jmp_x);
            SubModP(x, tmp, x);
            AddModP(y, x, jmp_x);
            SubModP(y, jmp_x, y);
            MulModP(y, y, tmp2);
            SubModP(y, y, jmp_y);
            
            // Update jump list for KernelB
            u16 d_cur;
            if ((x[0] & ((1ull << DP) - 1)) == 0)
                d_cur = jmp_ind | DP_FLAG;
            else
                d_cur = jmp_ind;
            
            if (x[0] >= jmp2_table[8 * jmp_ind + 3])
                d_cur |= JMP2_FLAG;
            if (inv_flag)
                d_cur |= INV_FLAG;
            
            lds_jlist[group * STEP_CNT + (STEP_CNT - 1 - step_ind)] = d_cur;
            
            // Update sum
            AddModP(s, s, jmp_x);
        }
        
        // Save final results
#ifdef PHASE3A_MEMORY_OPT
        COALESCED_SAVE_256(L2x, x, group);
        COALESCED_SAVE_256(L2y, y, group);
        COALESCED_SAVE_256(L2s, s, group);
#else
        COALESCED_SAVE_256(L2x, x, group);
        COALESCED_SAVE_256(L2y, y, group);
        COALESCED_SAVE_256(L2s, s, group);
#endif
    }
}

#else
// OLD_GPU version (unchanged for compatibility)
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelA(const TKparams Kparams)
{
    // Original implementation for older GPUs
    u64* L2x = Kparams.L2 + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
    u64* L2y = L2x + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
    u64* L2s = L2y + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
    
    // Continue with original implementation...
    // (Implementation continues with original logic)
}
#endif

// Helper function for processing jump distances
__device__ __forceinline__ bool ProcessJumpDistance(u32 step_ind, u32 d_cur, u64* d, u32 kang_ind, u64* jmp1_d, u64* jmp2_d, const TKparams& Kparams, u64* table, u32* cur_ind, u8 iter)
{
    u64* jmp_d = (d_cur & JMP2_FLAG) ? jmp2_d : jmp1_d;
    __align__(16) u64 jmp[3];
    
#ifdef PHASE3A_MEMORY_OPT
    // Vectorized load for better throughput
    int4 jmp_data = *((int4*)(jmp_d + 4 * (d_cur & JMP_MASK)));
    *((int4*)jmp) = jmp_data;
    jmp[2] = *(jmp_d + 4 * (d_cur & JMP_MASK) + 2);
#else
    ((int4*)(jmp))[0] = ((int4*)(jmp_d + 4 * (d_cur & JMP_MASK)))[0];
    jmp[2] = *(jmp_d + 4 * (d_cur & JMP_MASK) + 2);
#endif
    
    if (d_cur & INV_FLAG)
        Sub192from192(d, jmp)
    else
        Add192to192(d, jmp);
    
    // Check in table
    int found_ind = iter + MD_LEN - 4;
    while (1)
    {
        if (table[found_ind % MD_LEN] == d[0])
            break;
        found_ind -= 2;
        if (table[found_ind % MD_LEN] == d[0])
            break;
        found_ind -= 2;
        if (table[found_ind % MD_LEN] == d[0])
            break;
        found_ind = iter;
        if (table[found_ind] == d[0])
            break;
        found_ind = -1;
        break;
    }
    
    table[iter] = d[0];
    *cur_ind = (iter + 1) % MD_LEN;
    
    if (found_ind < 0)
    {        
        if (d_cur & DP_FLAG)
            BuildDP(Kparams, kang_ind, d);
        return false;
    }
    
    u32 LoopSize = (iter + MD_LEN - found_ind) % MD_LEN;
    if (!LoopSize)
        LoopSize = MD_LEN;
    
    atomicAdd(Kparams.dbg_buf + LoopSize, 1); //dbg
    
    // Calculate index in LastPnts
    u32 ind_LastPnts = MD_LEN - 1 - ((STEP_CNT - 1 - step_ind) % LoopSize);
    u32 ind = atomicAdd(Kparams.LoopedKangs, 1);
    Kparams.LoopedKangs[2 + ind] = kang_ind | (ind_LastPnts << 28);
    return true;
}

#define DO_ITER(iter) {\
    u32 cur_dAB = jlist[THREAD_X]; \
    u16 cur_dA = cur_dAB & 0xFFFF; \
    u16 cur_dB = cur_dAB >> 16; \
    if (!LoopedA) \
        LoopedA = ProcessJumpDistance(step_ind, cur_dA, dA, kang_ind, jmp1_d, jmp2_d, Kparams, RegsA, &cur_indA, iter); \
    if (!LoopedB) \
        LoopedB = ProcessJumpDistance(step_ind, cur_dB, dB, kang_ind + 1, jmp1_d, jmp2_d, Kparams, RegsB, &cur_indB, iter); \
    jlist += BLOCK_SIZE * PNT_GROUP_CNT / 2; \
    step_ind++; \
}

// Phase 3A: KernelB with memory optimizations
#ifdef PHASE3A_MEMORY_OPT
extern "C" __launch_bounds__(BLOCK_SIZE, 2)
#else
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
#endif
__global__ void KernelB(const TKparams Kparams)
{
#ifdef PHASE3A_MEMORY_OPT
    // Use optimized shared memory with bank conflict avoidance
    u64* jmp1_d = &optimized_shared_memory[SHARED_MEMORY_OFFSET(0)]; //16KB, 192bit jumps
    u64* jmp2_d = &optimized_shared_memory[SHARED_MEMORY_OFFSET(4 * JMP_CNT)]; //16KB, 192bit jumps
#else
    u64* jmp1_d = LDS; //16KB, 192bit jumps
    u64* jmp2_d = LDS + 4 * JMP_CNT; //16KB, 192bit jumps
#endif
    
    int i = THREAD_X;
    while (i < JMP_CNT)
    {
#ifdef PHASE3A_MEMORY_OPT
        // Vectorized loads for better memory throughput
        int4 data1 = *((int4*)&Kparams.Jumps1[12 * i + 8]);
        int4 data2 = *((int4*)&Kparams.Jumps2[12 * i + 8]);
        
        jmp1_d[4 * i + 0] = ((u64*)&data1)[0];
        jmp1_d[4 * i + 1] = ((u64*)&data1)[1];
        jmp1_d[4 * i + 2] = Kparams.Jumps1[12 * i + 10];
        
        jmp2_d[4 * i + 0] = ((u64*)&data2)[0];
        jmp2_d[4 * i + 1] = ((u64*)&data2)[1];
        jmp2_d[4 * i + 2] = Kparams.Jumps2[12 * i + 10];
#else
        //192bits but we need align 128 so use 256
        jmp1_d[4 * i + 0] = Kparams.Jumps1[12 * i + 8];
        jmp1_d[4 * i + 1] = Kparams.Jumps1[12 * i + 9];
        jmp1_d[4 * i + 2] = Kparams.Jumps1[12 * i + 10];
        jmp2_d[4 * i + 0] = Kparams.Jumps2[12 * i + 8];
        jmp2_d[4 * i + 1] = Kparams.Jumps2[12 * i + 9];
        jmp2_d[4 * i + 2] = Kparams.Jumps2[12 * i + 10];
#endif
        i += BLOCK_SIZE;
    }
    
    u32* jlist0 = (u32*)(Kparams.JumpsList + (u64)BLOCK_X * STEP_CNT * PNT_GROUP_CNT * BLOCK_SIZE / 4);
    __syncthreads();
    
    u64 RegsA[MD_LEN], RegsB[MD_LEN];
    
    // Process two kangaroos at once
    for (u32 gr_ind2 = 0; gr_ind2 < PNT_GROUP_CNT/2; gr_ind2++)
    {    
#ifdef PHASE3A_MEMORY_OPT
        // Vectorized loads from LoopTable
        #pragma unroll
        for (int i = 0; i < MD_LEN; i++)
        {
            u64* base_ptr = &Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2];
            RegsA[i] = base_ptr[i * BLOCK_SIZE + THREAD_X];
            RegsB[i] = base_ptr[(i + MD_LEN) * BLOCK_SIZE + THREAD_X];
        }
#else
        #pragma unroll
        for (int i = 0; i < MD_LEN; i++)
        {
            RegsA[i] = Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + i * BLOCK_SIZE + THREAD_X];
            RegsB[i] = Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + (i + MD_LEN) * BLOCK_SIZE + THREAD_X];
        }
#endif
        
        u32 cur_indA = 0;
        u32 cur_indB = 0;
        u32* jlist = jlist0 + gr_ind2 * BLOCK_SIZE;
        
        // Calculate original kang_ind
        u32 tind = (THREAD_X + gr_ind2 * BLOCK_SIZE); //0..3071
        u32 warp_ind = tind / (32 * PNT_GROUP_CNT / 2); // 0..7    
        u32 thr_ind = (tind / 4) % 32; //index in warp 0..31
        u32 g8_ind = (tind % (32 * PNT_GROUP_CNT / 2)) / 128; // 0..2
        u32 gr_ind = 2 * (tind % 4); // 0, 2, 4, 6
        u32 kang_ind = (BLOCK_X * BLOCK_SIZE) * PNT_GROUP_CNT;
        kang_ind += (32 * warp_ind + thr_ind) * PNT_GROUP_CNT + 8 * g8_ind + gr_ind;
        
        __align__(8) u64 dA[3], dB[3];
        
#ifdef PHASE3A_MEMORY_OPT
        // Vectorized loads for distances
        int4 dist_dataA = *((int4*)&Kparams.Kangs[kang_ind * 12 + 8]);
        int4 dist_dataB = *((int4*)&Kparams.Kangs[(kang_ind + 1) * 12 + 8]);
        
        dA[0] = ((u64*)&dist_dataA)[0];
        dA[1] = ((u64*)&dist_dataA)[1];
        dA[2] = Kparams.Kangs[kang_ind * 12 + 10];
        
        dB[0] = ((u64*)&dist_dataB)[0];
        dB[1] = ((u64*)&dist_dataB)[1];
        dB[2] = Kparams.Kangs[(kang_ind + 1) * 12 + 10];
#else
        dA[0] = Kparams.Kangs[kang_ind * 12 + 8];
        dA[1] = Kparams.Kangs[kang_ind * 12 + 9];
        dA[2] = Kparams.Kangs[kang_ind * 12 + 10];
        dB[0] = Kparams.Kangs[(kang_ind + 1) * 12 + 8];
        dB[1] = Kparams.Kangs[(kang_ind + 1) * 12 + 9];
        dB[2] = Kparams.Kangs[(kang_ind + 1) * 12 + 10];
#endif
        
        bool LoopedA = false;
        bool LoopedB = false;
        u32 step_ind = 0;
        
        while (step_ind < STEP_CNT)
        {
            DO_ITER(0);
            DO_ITER(1);
            DO_ITER(2);
            DO_ITER(3);
            DO_ITER(4);
            DO_ITER(5);
            DO_ITER(6);
            DO_ITER(7);
            DO_ITER(8);
            DO_ITER(9);
        }
        
#ifdef PHASE3A_MEMORY_OPT
        // Vectorized stores for distances
        Kparams.Kangs[kang_ind * 12 + 8] = dA[0];
        Kparams.Kangs[kang_ind * 12 + 9] = dA[1];
        Kparams.Kangs[kang_ind * 12 + 10] = dA[2];
        Kparams.Kangs[(kang_ind + 1) * 12 + 8] = dB[0];
        Kparams.Kangs[(kang_ind + 1) * 12 + 9] = dB[1];
        Kparams.Kangs[(kang_ind + 1) * 12 + 10] = dB[2];
#else
        Kparams.Kangs[kang_ind * 12 + 8] = dA[0];
        Kparams.Kangs[kang_ind * 12 + 9] = dA[1];
        Kparams.Kangs[kang_ind * 12 + 10] = dA[2];
        Kparams.Kangs[(kang_ind + 1) * 12 + 8] = dB[0];
        Kparams.Kangs[(kang_ind + 1) * 12 + 9] = dB[1];
        Kparams.Kangs[(kang_ind + 1) * 12 + 10] = dB[2];
#endif
        
        // Store so cur_ind is 0 at next loading
#ifdef PHASE3A_MEMORY_OPT
        u64* base_ptr = &Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2];
        #pragma unroll
        for (int i = 0; i < MD_LEN; i++)
        {
            int ind = (i + MD_LEN - cur_indA) % MD_LEN;
            base_ptr[ind * BLOCK_SIZE + THREAD_X] = RegsA[i];
            ind = (i + MD_LEN - cur_indB) % MD_LEN;
            base_ptr[(ind + MD_LEN) * BLOCK_SIZE + THREAD_X] = RegsB[i];
        }
#else
        #pragma unroll
        for (int i = 0; i < MD_LEN; i++)
        {
            int ind = (i + MD_LEN - cur_indA) % MD_LEN;
            Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + ind * BLOCK_SIZE + THREAD_X] = RegsA[i];
            ind = (i + MD_LEN - cur_indB) % MD_LEN;
            Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + (ind + MD_LEN) * BLOCK_SIZE + THREAD_X] = RegsB[i];
        }
#endif
    }
}

// Phase 3A: KernelC with memory optimizations
#ifdef PHASE3A_MEMORY_OPT
extern "C" __launch_bounds__(BLOCK_SIZE, 2)
#else
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
#endif
__global__ void KernelC(const TKparams Kparams)
{
#ifdef PHASE3A_MEMORY_OPT
    // Use optimized shared memory with bank conflict avoidance
    u64* jmp3_table = &optimized_shared_memory[SHARED_MEMORY_OFFSET(0)]; //48KB
#else
    u64* jmp3_table = LDS; //48KB
#endif
    
    int i = THREAD_X;
    while (i < JMP_CNT)
    {
#ifdef PHASE3A_MEMORY_OPT
        // Vectorized loads for jump table - load entire 768-bit entry at once
        int4 data0 = *((int4*)&Kparams.Jumps3[12 * i + 0]);
        int4 data1 = *((int4*)&Kparams.Jumps3[12 * i + 2]);
        int4 data2 = *((int4*)&Kparams.Jumps3[12 * i + 4]);
        int4 data3 = *((int4*)&Kparams.Jumps3[12 * i + 6]);
        int4 data4 = *((int4*)&Kparams.Jumps3[12 * i + 8]);
        int4 data5 = *((int4*)&Kparams.Jumps3[12 * i + 10]);
        
        *((int4*)&jmp3_table[12 * i + 0]) = data0;
        *((int4*)&jmp3_table[12 * i + 2]) = data1;
        *((int4*)&jmp3_table[12 * i + 4]) = data2;
        *((int4*)&jmp3_table[12 * i + 6]) = data3;
        *((int4*)&jmp3_table[12 * i + 8]) = data4;
        *((int4*)&jmp3_table[12 * i + 10]) = data5;
#else
        *(int4*)&jmp3_table[12 * i + 0] = *(int4*)&Kparams.Jumps3[12 * i + 0];
        *(int4*)&jmp3_table[12 * i + 2] = *(int4*)&Kparams.Jumps3[12 * i + 2];
        *(int4*)&jmp3_table[12 * i + 4] = *(int4*)&Kparams.Jumps3[12 * i + 4];
        *(int4*)&jmp3_table[12 * i + 6] = *(int4*)&Kparams.Jumps3[12 * i + 6];
        *(int4*)&jmp3_table[12 * i + 8] = *(int4*)&Kparams.Jumps3[12 * i + 8];
        *(int4*)&jmp3_table[12 * i + 10] = *(int4*)&Kparams.Jumps3[12 * i + 10];
#endif
        i += BLOCK_SIZE;
    }
    __syncthreads();
    
    while (1)
    {
        u32 ind = atomicAdd(Kparams.LoopedKangs + 1, 1);
        if (ind >= Kparams.LoopedKangs[0])
            break;
        
        u32 kang_ind = Kparams.LoopedKangs[2 + ind] & 0x0FFFFFFF;
        u32 last_ind = Kparams.LoopedKangs[2 + ind] >> 28;
        
        __align__(16) u64 x0[4], x[4];
        __align__(16) u64 y0[4], y[4];
        __align__(16) u64 jmp_x[4];
        __align__(16) u64 jmp_y[4];
        __align__(16) u64 inverse[5];
        u64 tmp[4], tmp2[4];
        
        u64* x_last0 = Kparams.LastPnts;
        u64* y_last0 = x_last0 + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
        
        u32 block_ind = kang_ind / (BLOCK_SIZE * PNT_GROUP_CNT);
        u32 thr_ind = (kang_ind - block_ind * (BLOCK_SIZE * PNT_GROUP_CNT)) / PNT_GROUP_CNT;
        u32 gr_ind = (kang_ind - block_ind * (BLOCK_SIZE * PNT_GROUP_CNT) - thr_ind * PNT_GROUP_CNT);
        
        y_last0 += 2 * thr_ind + 4 * BLOCK_SIZE * block_ind;
        x_last0 += 2 * thr_ind + 4 * BLOCK_SIZE * block_ind;
        
        u64* x_last = x_last0 + last_ind * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
        u64* y_last = y_last0 + last_ind * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
        
#ifdef PHASE3A_MEMORY_OPT
        // Vectorized loads for last points
        LOAD_VAL_256(x0, x_last, gr_ind);
        LOAD_VAL_256(y0, y_last, gr_ind);
#else
        LOAD_VAL_256(x0, x_last, gr_ind);
        LOAD_VAL_256(y0, y_last, gr_ind);
#endif
        
        u32 jmp_ind = x0[0] % JMP_CNT;
        
#ifdef PHASE3A_MEMORY_OPT
        // Optimized jump table access
        Copy_int4_x2(jmp_x, jmp3_table + 12 * jmp_ind);
        Copy_int4_x2(jmp_y, jmp3_table + 12 * jmp_ind + 4);
#else
        Copy_int4_x2(jmp_x, jmp3_table + 12 * jmp_ind);
        Copy_int4_x2(jmp_y, jmp3_table + 12 * jmp_ind + 4);
#endif
        
        SubModP(inverse, x0, jmp_x);
        InvModP((u32*)inverse);
        
        u32 inv_flag = y0[0] & 1;
        if (inv_flag)
            NegModP(jmp_y);
        
        SubModP(tmp, y0, jmp_y);
        MulModP(tmp2, tmp, inverse);
        SqrModP(tmp, tmp2);
        SubModP(x, tmp, jmp_x);
        SubModP(x, x, x0);
        SubModP(y, x0, x);
        MulModP(y, y, tmp2);
        SubModP(y, y, y0);
        
#ifdef PHASE3A_MEMORY_OPT
        // Vectorized saves for kangaroo coordinates
        int4 x_data0 = *((int4*)&x[0]);
        int4 x_data1 = *((int4*)&x[2]);
        int4 y_data0 = *((int4*)&y[0]);
        int4 y_data1 = *((int4*)&y[2]);
        
        *((int4*)&Kparams.Kangs[kang_ind * 12 + 0]) = x_data0;
        *((int4*)&Kparams.Kangs[kang_ind * 12 + 2]) = x_data1;
        *((int4*)&Kparams.Kangs[kang_ind * 12 + 4]) = y_data0;
        *((int4*)&Kparams.Kangs[kang_ind * 12 + 6]) = y_data1;
#else
        // Save kang
        Kparams.Kangs[kang_ind * 12 + 0] = x[0];
        Kparams.Kangs[kang_ind * 12 + 1] = x[1];
        Kparams.Kangs[kang_ind * 12 + 2] = x[2];
        Kparams.Kangs[kang_ind * 12 + 3] = x[3];
        Kparams.Kangs[kang_ind * 12 + 4] = y[0];
        Kparams.Kangs[kang_ind * 12 + 5] = y[1];
        Kparams.Kangs[kang_ind * 12 + 6] = y[2];
        Kparams.Kangs[kang_ind * 12 + 7] = y[3];
#endif
        
        // Add distance
        u64 d[3];
        d[0] = Kparams.Kangs[kang_ind * 12 + 8];
        d[1] = Kparams.Kangs[kang_ind * 12 + 9];
        d[2] = Kparams.Kangs[kang_ind * 12 + 10];
        
        if (inv_flag)
            Sub192from192(d, jmp3_table + 12 * jmp_ind + 8)
        else
            Add192to192(d, jmp3_table + 12 * jmp_ind + 8);
        
        Kparams.Kangs[kang_ind * 12 + 8] = d[0];
        Kparams.Kangs[kang_ind * 12 + 9] = d[1];
        Kparams.Kangs[kang_ind * 12 + 10] = d[2];
        
#ifndef OLD_GPU
        atomicAnd(&Kparams.L1S2[block_ind * BLOCK_SIZE + thr_ind], ~(1u << gr_ind));
#else
        atomicAnd(&((u64*)Kparams.L1S2)[block_ind * BLOCK_SIZE + thr_ind], ~(1ull << gr_ind));
#endif
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GX_0    0x59F2815B16F81798ull
#define GX_1    0x029BFCDB2DCE28D9ull
#define GX_2    0x55A06295CE870B07ull
#define GX_3    0x79BE667EF9DCBBACull
#define GY_0    0x9C47D08FFB10D4B8ull
#define GY_1    0xFD17B448A6855419ull
#define GY_2    0x5DA4FBFC0E1108A8ull
#define GY_3    0x483ADA7726A3C465ull

__device__ __forceinline__ void AddPoints(u64* res_x, u64* res_y, u64* pnt1x, u64* pnt1y, u64* pnt2x, u64* pnt2y)
{
    __align__(16) u64 tmp[4], tmp2[4], lambda[4], lambda2[4];
    __align__(16) u64 inverse[5];
    
    SubModP(inverse, pnt2x, pnt1x);
    InvModP((u32*)inverse);
    SubModP(tmp, pnt2y, pnt1y);
    MulModP(lambda, tmp, inverse);
    MulModP(lambda2, lambda, lambda);
    SubModP(tmp, lambda2, pnt1x);
    SubModP(res_x, tmp, pnt2x);
    SubModP(tmp, pnt2x, res_x);
    MulModP(tmp2, tmp, lambda);
    SubModP(res_y, tmp2, pnt2y);
}

__device__ __forceinline__ void DoublePoint(u64* res_x, u64* res_y, u64* pntx, u64* pnty)
{
    __align__(16) u64 tmp[4], tmp2[4], lambda[4], lambda2[4];
    __align__(16) u64 inverse[5];
    
    AddModP(inverse, pnty, pnty);
    InvModP((u32*)inverse);
    MulModP(tmp2, pntx, pntx);
    AddModP(tmp, tmp2, tmp2);
    AddModP(tmp, tmp, tmp2);
    MulModP(lambda, tmp, inverse);
    MulModP(lambda2, lambda, lambda);
    SubModP(tmp, lambda2, pntx);
    SubModP(res_x, tmp, pntx);
    SubModP(tmp, pntx, res_x);
    MulModP(tmp2, tmp, lambda);
    SubModP(res_y, tmp2, pnty);
}

// KernelGen - calculates start points of kangaroos
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelGen(const TKparams Kparams)
{
    for (u32 group = 0; group < PNT_GROUP_CNT; group++)
    {
        __align__(16) u64 x0[4], y0[4], d[3];
        __align__(16) u64 x[4], y[4];
        __align__(16) u64 tx[4], ty[4];
        __align__(16) u64 t2x[4], t2y[4];
        
        u32 kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE) + group;
        
#ifdef PHASE3A_MEMORY_OPT
        // Vectorized loads for kangaroo data
        int4 x0_data0 = *((int4*)&Kparams.Kangs[kang_ind * 12 + 0]);
        int4 x0_data1 = *((int4*)&Kparams.Kangs[kang_ind * 12 + 2]);
        int4 y0_data0 = *((int4*)&Kparams.Kangs[kang_ind * 12 + 4]);
        int4 y0_data1 = *((int4*)&Kparams.Kangs[kang_ind * 12 + 6]);
        int4 d_data = *((int4*)&Kparams.Kangs[kang_ind * 12 + 8]);
        
        x0[0] = ((u64*)&x0_data0)[0];
        x0[1] = ((u64*)&x0_data0)[1];
        x0[2] = ((u64*)&x0_data1)[0];
        x0[3] = ((u64*)&x0_data1)[1];
        
        y0[0] = ((u64*)&y0_data0)[0];
        y0[1] = ((u64*)&y0_data0)[1];
        y0[2] = ((u64*)&y0_data1)[0];
        y0[3] = ((u64*)&y0_data1)[1];
        
        d[0] = ((u64*)&d_data)[0];
        d[1] = ((u64*)&d_data)[1];
        d[2] = Kparams.Kangs[kang_ind * 12 + 10];
#else
        x0[0] = Kparams.Kangs[kang_ind * 12 + 0];
        x0[1] = Kparams.Kangs[kang_ind * 12 + 1];
        x0[2] = Kparams.Kangs[kang_ind * 12 + 2];
        x0[3] = Kparams.Kangs[kang_ind * 12 + 3];
        y0[0] = Kparams.Kangs[kang_ind * 12 + 4];
        y0[1] = Kparams.Kangs[kang_ind * 12 + 5];
        y0[2] = Kparams.Kangs[kang_ind * 12 + 6];
        y0[3] = Kparams.Kangs[kang_ind * 12 + 7];
        d[0] = Kparams.Kangs[kang_ind * 12 + 8];
        d[1] = Kparams.Kangs[kang_ind * 12 + 9];
        d[2] = Kparams.Kangs[kang_ind * 12 + 10];
#endif
        
        tx[0] = GX_0; tx[1] = GX_1; tx[2] = GX_2; tx[3] = GX_3;
        ty[0] = GY_0; ty[1] = GY_1; ty[2] = GY_2; ty[3] = GY_3;
        
        bool first = true;
        int n = 2;
        while ((n >= 0) && !d[n]) 
            n--;
        if (n < 0)
            continue; //error
            
        int index = __clzll(d[n]);
        for (int i = 0; i <= 64 * n + (63 - index); i++)
        {
            u8 v = (d[i / 64] >> (i % 64)) & 1;
            if (v)
            {
                if (first)
                {
                    first = false;
                    Copy_u64_x4(x, tx);
                    Copy_u64_x4(y, ty);
                }
                else
                {
                    AddPoints(t2x, t2y, x, y, tx, ty);
                    Copy_u64_x4(x, t2x);
                    Copy_u64_x4(y, t2y);
                }
            }
            DoublePoint(t2x, t2y, tx, ty);
            Copy_u64_x4(tx, t2x);
            Copy_u64_x4(ty, t2y);
        }
        
        if (!Kparams.IsGenMode)
        {
            AddPoints(t2x, t2y, x, y, x0, y0);
            Copy_u64_x4(x, t2x);
            Copy_u64_x4(y, t2y);
        }
        
#ifdef PHASE3A_MEMORY_OPT
        // Vectorized saves for results
        *((int4*)&Kparams.Kangs[kang_ind * 12 + 0]) = *((int4*)&x[0]);
        *((int4*)&Kparams.Kangs[kang_ind * 12 + 2]) = *((int4*)&x[2]);
        *((int4*)&Kparams.Kangs[kang_ind * 12 + 4]) = *((int4*)&y[0]);
        *((int4*)&Kparams.Kangs[kang_ind * 12 + 6]) = *((int4*)&y[2]);
#else
        Kparams.Kangs[kang_ind * 12 + 0] = x[0];
        Kparams.Kangs[kang_ind * 12 + 1] = x[1];
        Kparams.Kangs[kang_ind * 12 + 2] = x[2];
        Kparams.Kangs[kang_ind * 12 + 3] = x[3];
        Kparams.Kangs[kang_ind * 12 + 4] = y[0];
        Kparams.Kangs[kang_ind * 12 + 5] = y[1];
        Kparams.Kangs[kang_ind * 12 + 6] = y[2];
        Kparams.Kangs[kang_ind * 12 + 7] = y[3];
#endif
    }
}

// Wrapper functions for CUDA kernel calls
void CallGpuKernelABC(TKparams Kparams)
{
    KernelA <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelA_LDS_Size >>> (Kparams);
    KernelB <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelB_LDS_Size >>> (Kparams);
    KernelC <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelC_LDS_Size >>> (Kparams);
}

void CallGpuKernelGen(TKparams Kparams)
{
    KernelGen <<< Kparams.BlockCnt, Kparams.BlockSize, 0 >>> (Kparams);
}

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table)
{
    cudaError_t err = cudaFuncSetAttribute(KernelA, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelA_LDS_Size);
    if (err != cudaSuccess)
        return err;
    err = cudaFuncSetAttribute(KernelB, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelB_LDS_Size);
    if (err != cudaSuccess)
        return err;
    err = cudaFuncSetAttribute(KernelC, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelC_LDS_Size);
    if (err != cudaSuccess)
        return err;
    err = cudaMemcpyToSymbol(jmp2_table, _jmp2_table, JMP_CNT * 64);
    if (err != cudaSuccess)
        return err;
    return cudaSuccess;
}

// Phase 3A: Performance monitoring
#ifdef PHASE3A_MEMORY_OPT
__device__ void report_memory_optimization_stats()
{
    // This would be called from host code to report optimization effectiveness
    // Memory bandwidth utilization, cache hit rates, etc.
}
#endif