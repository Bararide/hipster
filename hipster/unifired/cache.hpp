#ifndef HIPSTER_UNIFIRED_CACHE_HPP
#define HIPSTER_UNIFIRED_CACHE_HPP

namespace hipster {

[[nodiscard]] inline bool has_clwb() noexcept {
  unsigned eax, ebx, ecx, edx;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx >> 24) & 1u;
}

[[nodiscard]] inline bool has_clflushopt() noexcept {
  unsigned eax, ebx, ecx, edx;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx >> 23) & 1u;
}

[[nodiscard]] inline bool has_avx512f() noexcept {
  unsigned eax, ebx, ecx, edx;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx >> 16) & 1u;
}

struct Caps {
  bool clwb;
  bool clflushopt;
  bool avx512f;
  
  static const Caps &get() noexcept {
    static const Caps c{has_clwb(), has_clflushopt(), has_avx512f()};
    return c;
  }
};

struct FlushPolicyClwb {};
struct FlushPolicyClflushopt {};
struct FlushPolicyNone {};
struct FlushPolicyAuto {};

template <typename Policy> struct CacheFlush;

template <> struct CacheFlush<FlushPolicyClwb> {
  __attribute__((target("clwb"))) static void flush(char *p,
                                                    size_t len) noexcept {
    for (size_t i = 0; i < len; i += 64) {
      _mm_clwb(p + i);
    }
    _mm_sfence();
  }
};

template <> struct CacheFlush<FlushPolicyClflushopt> {
  __attribute__((target("clflushopt"))) static void flush(char *p,
                                                          size_t len) noexcept {
    for (size_t i = 0; i < len; i += 64) {
      _mm_clflushopt(p + i);
    }
    _mm_sfence();
  }
};

template <> struct CacheFlush<FlushPolicyNone> {
  static void flush(char *, size_t) noexcept { _mm_sfence(); }
};

template <> struct CacheFlush<FlushPolicyAuto> {
  __attribute__((target("clwb,clflushopt"))) static void
  flush(char *p, size_t len) noexcept {
    const auto &caps = Caps::get();
    if (caps.clwb) {
      for (size_t i = 0; i < len; i += 64) {
        _mm_clwb(p + i);
      }
    } else if (caps.clflushopt) {
      for (size_t i = 0; i < len; i += 64) {
        _mm_clflushopt(p + i);
      }
    }
    _mm_sfence();
  }
};

inline void prefetchWrite(const char *p, size_t len) noexcept {
  static const bool is_amd = [] {
    unsigned int eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);
    return ebx == 0x68747541 && ecx == 0x444d4163 && edx == 0x69746e65;
  }();

  for (size_t i = 0; i < len; i += 64) {
    if (is_amd) {
      asm volatile("prefetchw (%0)" : : "r"(p + i) : "memory");
    } else {
      __builtin_prefetch(p + i, 1, 1);
    }
  }
}

} // namespace hipster

#endif // HIPSTER_UNIFIRED_CACHE_HPP