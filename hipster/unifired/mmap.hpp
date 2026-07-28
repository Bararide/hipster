#ifndef HIPSTER_UNIFIRED_MMAP_HPP
#define HIPSTER_UNIFIRED_MMAP_HPP

#include <concepts>
#include <linux/memfd.h>
#include <sys/mman.h>

namespace hipster {

struct MmapConfig {
  unsigned int memfd_flags = 0;
  int prot = 0;
  int map_flags = 0;
  int madvise = 0;
};

template <typename T>
concept IsMmapOption = requires(T t, MmapConfig &cfg) {
  { t.apply(cfg) } -> std::same_as<void>;
};

struct MemfdCloexec {
  void apply(MmapConfig &cfg) const noexcept { cfg.memfd_flags |= MFD_CLOEXEC; }
};

struct MemfdAllowSealing {
  void apply(MmapConfig &cfg) const noexcept {
    cfg.memfd_flags |= MFD_ALLOW_SEALING;
  }
};

struct ProtRead {
  void apply(MmapConfig &cfg) const noexcept { cfg.prot |= PROT_READ; }
};

struct ProtWrite {
  void apply(MmapConfig &cfg) const noexcept { cfg.prot |= PROT_WRITE; }
};

struct ProtNone {
  void apply(MmapConfig &cfg) const noexcept { cfg.prot = PROT_NONE; }
};

struct MapShared {
  void apply(MmapConfig &cfg) const noexcept {
    cfg.map_flags = (cfg.map_flags & ~MAP_PRIVATE) | MAP_SHARED;
  }
};

struct MapPrivate {
  void apply(MmapConfig &cfg) const noexcept {
    cfg.map_flags = (cfg.map_flags & ~MAP_SHARED) | MAP_PRIVATE;
  }
};

struct MapPopulate {
  void apply(MmapConfig &cfg) const noexcept { cfg.map_flags |= MAP_POPULATE; }
};

struct MapHugeTlb {
  void apply(MmapConfig &cfg) const noexcept { cfg.map_flags |= MAP_HUGETLB; }
};

struct MapNoHugeTlb {
  void apply(MmapConfig &cfg) const noexcept { cfg.map_flags &= ~MAP_HUGETLB; }
};

struct AdviseSequential {
  void apply(MmapConfig &cfg) const noexcept { cfg.madvise |= MADV_SEQUENTIAL; }
};

struct AdviseRandom {
  void apply(MmapConfig &cfg) const noexcept {
    cfg.madvise = (cfg.madvise & ~MADV_SEQUENTIAL) | MADV_RANDOM;
  }
};

struct AdviseDontFork {
  void apply(MmapConfig &cfg) const noexcept { cfg.madvise |= MADV_DONTFORK; }
};

struct AdviseNoDontFork {
  void apply(MmapConfig &cfg) const noexcept { cfg.madvise &= ~MADV_DONTFORK; }
};

struct DefaultDmaConfig {
  void apply(MmapConfig &cfg) const noexcept {
    cfg.memfd_flags = MFD_CLOEXEC | MFD_ALLOW_SEALING;
    cfg.prot = PROT_READ | PROT_WRITE;
    cfg.map_flags = MAP_SHARED | MAP_POPULATE;
    cfg.madvise = MADV_SEQUENTIAL | MADV_DONTFORK;
  }
};

} // namespace hipster

#endif // HIPSTER_UNIFIRED_MMAP_HPP