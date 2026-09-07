#include "moe_handoff.h"
#include <thread>

// Compact v1.4.7-class GPU/CPU handoff TU for host aarch64-stub tests.
// The production pin ships the full source; this fixture only needs the
// unguarded x86 pause builtin that nvcc rejects on aarch64.

inline void cpu_pause_()
{
#ifdef __linux__
    __builtin_ia32_pause();
#else
    _mm_pause();
#endif
}

void exl3_moe_cpu_worker_run(uintptr_t)
{
    cpu_pause_();
}
