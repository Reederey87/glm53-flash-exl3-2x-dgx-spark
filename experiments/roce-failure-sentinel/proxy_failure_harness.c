// Include the exact installed source via -I; no RDMA context or GPU is created.
#include "_roce_proxy.c"

int main(void) {
  if (roce_abi_version() != 8) {
    fprintf(stderr, "expected candidate proxy ABI 8\n");
    return 1;
  }
  const uint32_t sequences[] = {0u, 1u, UINT32_MAX};
  int cases = 0;
  for (int operation = 0; operation < 2; operation++) {
    for (int index = 0; index < 3; index++) {
      for (int failed = 0; failed < 2; failed++) {
        uint32_t control[ROCE_CTRL_BYTES / sizeof(uint32_t)] = {0};
        roce_ctx_t context = {0};
        context.world = 2;
        context.rank = 0;
        context.region = (uint8_t*)control;
        context.region_bytes = sizeof(control);
        context.pipe_bytes = 4096;
        context.varlen_bytes = 4096;
        // No HCA, QP or posting: every ready word precedes the tested sequence.
        // running=0 bounds both old and new predicates to one idle-check cycle.
        atomic_init(&context.running, 0);
        atomic_init(&context.failed, 0);
        uint32_t seq = sequences[index];
        control[CTRL_READY_RS] = seq - 1u;
        control[CTRL_READY_AG] = seq - 1u;
        control[CTRL_READY_VL] = seq - 1u;
        control[CTRL_ERROR] = seq;
        control[CTRL_FAILED] = (uint32_t)failed;
        int result = operation == 0 ? run_pipe_op(&context, seq, 32, 1, 1)
                                    : run_varlen_op(&context, seq, 16, 1, 1);
        const char* expected = failed ? "abandoned" : "stopped inside";
        if (result != -1 || context.ops_posted != 0 || strstr(context.err, expected) == NULL) {
          fprintf(stderr, "op=%d seq=%u failed=%d result=%d error=%s\n",
                  operation, seq, failed, result, context.err);
          return 1;
        }
        cases++;
      }
    }
  }
  printf("proxy_failure_cases_passed=%d\n", cases);
  return 0;
}
