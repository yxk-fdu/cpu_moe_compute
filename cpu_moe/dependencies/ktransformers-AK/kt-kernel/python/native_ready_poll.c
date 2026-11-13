
#include <stdint.h>
#include <stdatomic.h>
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#elif !defined(__aarch64__)
#include <sched.h>
#endif
static inline void poll_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    sched_yield();
#endif
}
#define N 2048
static struct { _Atomic uint32_t *ptr; uint32_t seen; } slots[N];
static _Atomic unsigned count;
static uint64_t now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
int register_flag(void *p){unsigned i=atomic_load(&count);if(i>=N)return -1;slots[i].ptr=p;slots[i].seen=0;atomic_store_explicit(&count,i+1,memory_order_release);return i;}
int poll_next(uint64_t *stamp,uint32_t *seq){uint64_t end=now_ns()+1000000;do{unsigned n=atomic_load_explicit(&count,memory_order_acquire);for(unsigned i=0;i<n;i++){uint32_t v=atomic_load_explicit(slots[i].ptr,memory_order_acquire);if(v!=slots[i].seen){if(v!=slots[i].seen+1)return -2;slots[i].seen=v;atomic_thread_fence(memory_order_acquire);*stamp=now_ns();*seq=v;return i;}}poll_relax();}while(now_ns()<end);return -1;}
