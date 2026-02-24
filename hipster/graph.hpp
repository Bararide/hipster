#ifndef HIPSTER_GRAPH_HPP
#define HIPSTER_GRAPH_HPP

#include "alies.hpp"

namespace hipster {

class HipGraphNode;
class HipGraphExec;

class HipGraphNode {
public:
  HipGraphNode() = default;
  explicit HipGraphNode(hipGraphNode_t node) : node_(node) {}

  hipGraphNode_t get() const { return node_; }
  operator hipGraphNode_t() const { return node_; }

  hipGraphNodeType getType() const {
    hipGraphNodeType type;
    hipError_t err = hipGraphNodeGetType(node_, &type);
    if (err != hipSuccess) {
      std::cerr << "hipGraphNodeGetType error: " << hipGetErrorString(err)
                << std::endl;
      return hipGraphNodeTypeEmpty;
    }
    return type;
  }

  bool valid() const { return node_ != nullptr; }

private:
  hipGraphNode_t node_ = nullptr;
};

class HipGraphExec {
public:
  explicit HipGraphExec(hipGraphExec_t exec) : exec_(exec) {}

  ~HipGraphExec() {
    if (exec_) {
      hipError_t err = hipGraphExecDestroy(exec_);
      if (err != hipSuccess) {
        std::cerr << "hipGraphExecDestroy error: " << hipGetErrorString(err)
                  << std::endl;
      }
    }
  }

  HipGraphExec(const HipGraphExec &) = delete;
  HipGraphExec &operator=(const HipGraphExec &) = delete;

  HipGraphExec(HipGraphExec &&other) noexcept {
    exec_ = other.exec_;
    other.exec_ = nullptr;
  }

  HipGraphExec &operator=(HipGraphExec &&other) noexcept {
    hipError_t err = hipSuccess;

    if (this != &other) {
      if (exec_) {
        err = hipGraphExecDestroy(exec_);
      }

      if (err != hipSuccess) {
        std::cerr << "hipGraphExecDestroy error: " << hipGetErrorString(err)
                  << std::endl;
      }

      exec_ = other.exec_;
      other.exec_ = nullptr;
    }
    return *this;
  }

  void launch(hipStream_t stream) const {
    if (!exec_)
      return;
    hipError_t err = hipGraphLaunch(exec_, stream);
    if (err != hipSuccess) {
      std::cerr << "hipGraphLaunch error: " << hipGetErrorString(err)
                << std::endl;
    }
  }

  void updateKernelNode(const HipGraphNode &node, dim3 gridDim, dim3 blockDim,
                        void **args, size_t sharedMemBytes = 0) {
    if (!exec_ || !node.valid())
      return;

    hipKernelNodeParams params{};
    params.gridDim = gridDim;
    params.blockDim = blockDim;
    params.sharedMemBytes = sharedMemBytes;
    params.kernelParams = args;
    params.extra = nullptr;

    hipError_t err =
        hipGraphExecKernelNodeSetParams(exec_, node.get(), &params);
    if (err != hipSuccess) {
      std::cerr << "hipGraphExecKernelNodeSetParams error: "
                << hipGetErrorString(err) << std::endl;
    }
  }

  hipGraphExec_t get() const { return exec_; }
  operator hipGraphExec_t() const { return exec_; }

  bool valid() const { return exec_ != nullptr; }

private:
  hipGraphExec_t exec_ = nullptr;
};

class HipGraph {
public:
  HipGraph() {
    hipError_t err = hipGraphCreate(&graph_, 0);
    if (err != hipSuccess) {
      std::cerr << "hipGraphCreate error: " << hipGetErrorString(err)
                << std::endl;
      graph_ = nullptr;
    }
  }

  ~HipGraph() {
    if (graph_) {
      hipError_t err = hipGraphDestroy(graph_);
      if (err != hipSuccess) {
        std::cerr << "hipGraphDestroy error: " << hipGetErrorString(err)
                  << std::endl;
      }
    }
  }

  HipGraph(const HipGraph &) = delete;
  HipGraph &operator=(const HipGraph &) = delete;

  HipGraph(HipGraph &&other) noexcept {
    graph_ = other.graph_;
    other.graph_ = nullptr;
  }

  HipGraph &operator=(HipGraph &&other) noexcept {
    hipError_t err = hipSuccess;

    if (this != &other) {
      if (graph_) {
        err = hipGraphDestroy(graph_);
      }
      if (err != hipSuccess) {
        std::cerr << "hipGraphExecDestroy error: " << hipGetErrorString(err)
                  << std::endl;
      }

      graph_ = other.graph_;
      other.graph_ = nullptr;
    }
    return *this;
  }

  HipGraphNode addKernelNode(const void *kernel,
                             const std::vector<HipGraphNode> &dependencies,
                             dim3 gridDim, dim3 blockDim, void **args,
                             size_t sharedMemBytes = 0) {
    hipKernelNodeParams params{};
    params.func = const_cast<void *>(kernel);
    params.gridDim = gridDim;
    params.blockDim = blockDim;
    params.sharedMemBytes = sharedMemBytes;
    params.kernelParams = args;
    params.extra = nullptr;

    std::vector<hipGraphNode_t> depNodes;
    depNodes.reserve(dependencies.size());
    for (const auto &node : dependencies) {
      depNodes.push_back(node.get());
    }

    hipGraphNode_t node{};
    hipError_t err = hipGraphAddKernelNode(
        &node, graph_, depNodes.empty() ? nullptr : depNodes.data(),
        static_cast<int>(depNodes.size()), &params);
    if (err != hipSuccess) {
      std::cerr << "hipGraphAddKernelNode error: " << hipGetErrorString(err)
                << std::endl;
      return HipGraphNode();
    }

    return HipGraphNode(node);
  }

  HipGraphNode addMemcpyNode(const std::vector<HipGraphNode> &dependencies,
                             void *dst, const void *src, size_t count,
                             hipMemcpyKind kind) {
    std::vector<hipGraphNode_t> depNodes;
    depNodes.reserve(dependencies.size());
    for (const auto &node : dependencies) {
      depNodes.push_back(node.get());
    }

    hipGraphNode_t node{};
    hipError_t err = hipGraphAddMemcpyNode1D(
        &node, graph_, depNodes.empty() ? nullptr : depNodes.data(),
        static_cast<int>(depNodes.size()), dst, src, count, kind);
    if (err != hipSuccess) {
      std::cerr << "hipGraphAddMemcpyNode1D error: " << hipGetErrorString(err)
                << std::endl;
      return HipGraphNode();
    }

    return HipGraphNode(node);
  }

  HipGraphNode addMemsetNode(const std::vector<HipGraphNode> &dependencies,
                             void *dst, int value, size_t count) {
    hipMemsetParams params{};
    params.dst = dst;
    params.value = value;
    params.pitch = 0;
    params.elementSize = 1;
    params.width = count;
    params.height = 1;

    std::vector<hipGraphNode_t> depNodes;
    depNodes.reserve(dependencies.size());
    for (const auto &node : dependencies) {
      depNodes.push_back(node.get());
    }

    hipGraphNode_t node{};
    hipError_t err = hipGraphAddMemsetNode(
        &node, graph_, depNodes.empty() ? nullptr : depNodes.data(),
        static_cast<int>(depNodes.size()), &params);
    if (err != hipSuccess) {
      std::cerr << "hipGraphAddMemsetNode error: " << hipGetErrorString(err)
                << std::endl;
      return HipGraphNode();
    }

    return HipGraphNode(node);
  }

  HipGraphNode addEventWaitNode(const std::vector<HipGraphNode> &dependencies,
                                hipEvent_t event) {
    std::vector<hipGraphNode_t> depNodes;
    depNodes.reserve(dependencies.size());
    for (const auto &node : dependencies) {
      depNodes.push_back(node.get());
    }

    hipGraphNode_t node{};
    hipError_t err = hipGraphAddEventWaitNode(
        &node, graph_, depNodes.empty() ? nullptr : depNodes.data(),
        static_cast<int>(depNodes.size()), event);
    if (err != hipSuccess) {
      std::cerr << "hipGraphAddEventWaitNode error: " << hipGetErrorString(err)
                << std::endl;
      return HipGraphNode();
    }

    return HipGraphNode(node);
  }

  HipGraphNode addEventRecordNode(const std::vector<HipGraphNode> &dependencies,
                                  hipEvent_t event) {
    std::vector<hipGraphNode_t> depNodes;
    depNodes.reserve(dependencies.size());
    for (const auto &node : dependencies) {
      depNodes.push_back(node.get());
    }

    hipGraphNode_t node{};
    hipError_t err = hipGraphAddEventRecordNode(
        &node, graph_, depNodes.empty() ? nullptr : depNodes.data(),
        static_cast<int>(depNodes.size()), event);
    if (err != hipSuccess) {
      std::cerr << "hipGraphAddEventRecordNode error: "
                << hipGetErrorString(err) << std::endl;
      return HipGraphNode();
    }

    return HipGraphNode(node);
  }

  HipGraphExec instantiate() {
    hipGraphExec_t exec{};
    hipError_t err = hipGraphInstantiate(&exec, graph_, nullptr, nullptr, 0);
    if (err != hipSuccess) {
      std::cerr << "hipGraphInstantiate error: " << hipGetErrorString(err)
                << std::endl;
      return HipGraphExec(nullptr);
    }
    return HipGraphExec(exec);
  }

  hipGraph_t get() const { return graph_; }
  operator hipGraph_t() const { return graph_; }

  bool valid() const { return graph_ != nullptr; }

private:
  hipGraph_t graph_ = nullptr;
};

} // namespace hipster

#endif // HIPSTER_GRAPH_HPP