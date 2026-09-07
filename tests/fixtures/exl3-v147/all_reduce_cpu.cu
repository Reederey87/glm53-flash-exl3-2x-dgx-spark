#include "all_reduce_cpu_avx2.h"
#include <thread>

// Compact v1.4.7-class CPU all-reduce TU for host aarch64-stub tests.
// The production pin ships the full source; this fixture only needs the
// unguarded x86 pause builtin that nvcc rejects on aarch64.

void reduce_wait()
{
#ifdef __linux__
    __builtin_ia32_pause();
#else
    _mm_pause();
#endif
}
