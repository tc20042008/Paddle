#include "paddle/cinn/frontend/dag_generator.h"
#include <random>

namespace cinn::frontend {

namespace {

bool IsValidNumSourceTensorsImpl(
    const Nope<std::monostate>&,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors,
    const DAGGenRequirement& requirement) {
  return true;
}

bool IsValidNumSourceTensorsImpl(
    const AddSinkTensor<std::monostate>&,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors,
    const DAGGenRequirement& requirement) {
  return num_source_tensors <= requirement.max_width;
}

bool IsValidNumSourceTensorsImpl(
    const AddUnaryUpstreamOp<std::monostate>&,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors,
    const DAGGenRequirement& requirement) {
  return true;
}

bool IsValidNumSourceTensorsImpl(
    const AddBinaryUpstreamOp<std::monostate>&,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors,
    const DAGGenRequirement& requirement) {
  return num_source_tensors <= requirement.max_width;
}

bool IsValidNumSourceTensorsImpl(
    const InsertBinaryUpstreamOp<std::monostate>&,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors,
    const DAGGenRequirement& requirement) {
  return num_source_tensors <= requirement.max_width;
}

bool IsValidNumSourceTensorsImpl(
    const AddBinaryCloneUpstream<std::monostate>&,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors,
    const DAGGenRequirement& requirement) {
  return num_core_source_tensors < num_source_tensors;
}

bool IsValidNumSourceTensorsImpl(
    const MarkFinalSourceTensor<std::monostate>&,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors,
    const DAGGenRequirement& requirement) {
  return num_core_source_tensors < num_source_tensors;
}

bool IsValidNumSourceTensors(
    const IrGenType<std::monostate>& type,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors,
    const DAGGenRequirement& requirement) {
  return std::visit([&](const auto& impl) {
    return IsValidNumSourceTensorsImpl(impl, num_core_source_tensors, num_source_tensors, requirement);
  }, type);
}

bool IsValidSourceTensorIndexImpl(
    const NopeGenInstruction& instruction,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  return true;
}

bool IsValidSourceTensorIndexImpl(
    const AddSinkTensorGenInstruction& instruction,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  return true;
}

bool IsValidSourceTensorIndexImpl(
    const AddUnaryUpstreamOpGenInstruction& instruction,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  return (instruction.source_tensor_index >= num_core_source_tensors)
    && (instruction.source_tensor_index < num_source_tensors);
}

bool IsValidSourceTensorIndexImpl(
    const AddBinaryUpstreamOpGenInstruction& instruction,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  return (instruction.source_tensor_index >= num_core_source_tensors)
    && (instruction.source_tensor_index < num_source_tensors);
}

bool IsValidSourceTensorIndexImpl(
    const InsertBinaryUpstreamOpGenInstruction& instruction,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  return (instruction.source_tensor_index >= 0)
    && (instruction.source_tensor_index < num_core_source_tensors);
}

bool IsValidSourceTensorIndexImpl(
    const AddBinaryCloneUpstreamGenInstruction& instruction,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  return (instruction.lhs_source_tensor_index >= 0)
    && (instruction.lhs_source_tensor_index < num_source_tensors)
    && (instruction.rhs_source_tensor_index >= num_core_source_tensors)
    && (instruction.rhs_source_tensor_index < num_source_tensors);
}

bool IsValidSourceTensorIndexImpl(
    const MarkFinalSourceTensorGenInstruction& instruction,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  return (instruction.source_tensor_index >= num_core_source_tensors)
    && (instruction.source_tensor_index < num_source_tensors);
}

bool IsValidSourceTensorIndex(
    const DAGGenInstruction& instruction,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  return std::visit([&](const auto& impl) {
    return IsValidSourceTensorIndexImpl(impl, num_core_source_tensors, num_source_tensors);
  }, instruction.variant());
}

void UpdateNumSourceTensorsImpl(
    int64_t* num_source_tensors, const NopeGenInstruction& instruction) {
  // Do nothing
}

void UpdateNumSourceTensorsImpl(
    int64_t* num_source_tensors,
    const AddSinkTensorGenInstruction& instruction) {
  ++*num_source_tensors;
}

void UpdateNumSourceTensorsImpl(
    int64_t* num_source_tensors,
    const AddUnaryUpstreamOpGenInstruction& instruction) {
  // Do nothing
}

void UpdateNumSourceTensorsImpl(
  int64_t* num_source_tensors,
  const AddBinaryUpstreamOpGenInstruction& instruction) {
  ++*num_source_tensors;
}

void UpdateNumSourceTensorsImpl(
  int64_t* num_source_tensors,
  const InsertBinaryUpstreamOpGenInstruction& instruction) {
  ++*num_source_tensors;
}

void UpdateNumSourceTensorsImpl(
  int64_t* num_source_tensors,
  const AddBinaryCloneUpstreamGenInstruction& instruction) {
  --*num_source_tensors;
}

void UpdateNumSourceTensorsImpl(
  int64_t* num_source_tensors,
  const MarkFinalSourceTensorGenInstruction& instruction) {
  --*num_source_tensors;
}

void UpdateNumSourceTensors(
    int64_t* num_source_tensors,
    const DAGGenInstruction& instruction) {
  std::visit([&](const auto& impl) {
    return UpdateNumSourceTensorsImpl(num_source_tensors, impl);
  }, instruction.variant());
}

int64_t GetRandomInt(int64_t start, int64_t end) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int64_t> dis(start, end);
  return dis(mt);
}

const DAGGenInstruction GenerateDAGGenInstructionImpl(
    const Nope<std::monostate>& type, 
    const DAGGenRequirement& requirement,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  return NopeGenInstruction{};
}

const DAGGenInstruction GenerateDAGGenInstructionImpl(
    const AddSinkTensor<std::monostate>& type, 
    const DAGGenRequirement& requirement,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  return AddSinkTensorGenInstruction{};
}

const DAGGenInstruction GenerateDAGGenInstructionImpl(
    const AddUnaryUpstreamOp<std::monostate>& type, 
    const DAGGenRequirement& requirement,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  PADDLE_ENFORCE_LT(num_core_source_tensors, num_source_tensors);
  int64_t random_int =
    GetRandomInt(num_core_source_tensors, num_source_tensors - 1);
  return AddUnaryUpstreamOpGenInstruction{
    .source_tensor_index=random_int,
    .dag_tag=requirement.dag_tag,
  };
}

const DAGGenInstruction GenerateDAGGenInstructionImpl(
    const AddBinaryUpstreamOp<std::monostate>& type, 
    const DAGGenRequirement& requirement,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  PADDLE_ENFORCE_LT(num_core_source_tensors, num_source_tensors);
  int64_t random_int =
    GetRandomInt(num_core_source_tensors, num_source_tensors - 1);
  return AddBinaryUpstreamOpGenInstruction{
    .source_tensor_index=random_int,
    .dag_tag=requirement.dag_tag,
  };
}

const DAGGenInstruction GenerateDAGGenInstructionImpl(
    const InsertBinaryUpstreamOp<std::monostate>& type, 
    const DAGGenRequirement& requirement,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  int64_t random_int = GetRandomInt(0, num_core_source_tensors - 1);
  return InsertBinaryUpstreamOpGenInstruction{
    .source_tensor_index=random_int,
    .dag_tag=requirement.dag_tag,
  };
}
const DAGGenInstruction GenerateDAGGenInstructionImpl(
    const AddBinaryCloneUpstream<std::monostate>& type, 
    const DAGGenRequirement& requirement,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  PADDLE_ENFORCE_LT(num_core_source_tensors, num_source_tensors);
  int64_t lhs_random_int =
    GetRandomInt(0, num_source_tensors - 1);
  int64_t rhs_random_int =
    GetRandomInt(num_core_source_tensors, num_source_tensors - 1);
  return AddBinaryCloneUpstreamGenInstruction{
    .lhs_source_tensor_index=lhs_random_int,
    .rhs_source_tensor_index=rhs_random_int,
  };
}

const DAGGenInstruction GenerateDAGGenInstructionImpl(
    const MarkFinalSourceTensor<std::monostate>& type, 
    const DAGGenRequirement& requirement,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  PADDLE_ENFORCE_LT(num_core_source_tensors, num_source_tensors);
  int64_t random_int =
    GetRandomInt(num_core_source_tensors, num_source_tensors - 1);
  return MarkFinalSourceTensorGenInstruction{
    .source_tensor_index=random_int,
  };
}

const DAGGenInstruction GenerateDAGGenInstruction(
    const IrGenType<std::monostate>& type, 
    const DAGGenRequirement& requirement,
    int64_t num_core_source_tensors,
    int64_t num_source_tensors) {
  return std::visit([&](const auto& impl) {
    return GenerateDAGGenInstructionImpl(
        impl, requirement, num_core_source_tensors, num_source_tensors);
  }, type);
}

class ConstDAGGenInstructions {
 public:
  ConstDAGGenInstructions(
      const adt::Stack<const DAGGenInstruction>& instructions)
    : instructions_(instructions),
      current_num_source_tensors_(0) {}
  ConstDAGGenInstructions(const ConstDAGGenInstructions&) = delete;
  ConstDAGGenInstructions(ConstDAGGenInstructions&&) = delete;

  const adt::Stack<const DAGGenInstruction>& instructions() const {
    return instructions_;
  }

  void TryPop() {
    if (instructions_.empty()) return;
    const auto& instruction = instructions_.top();
    UpdateNumSourceTensors(&current_num_source_tensors_, instruction);
    instructions_.pop();
  }

  int64_t current_num_source_tensors() const {
    return current_num_source_tensors_;
  }

 protected:
  adt::Stack<const DAGGenInstruction> instructions_;
  int64_t current_num_source_tensors_;
};

class MutDAGGenInstructions {
 public:
  MutDAGGenInstructions()
    : instructions_(), current_num_source_tensors_(0) {}
  MutDAGGenInstructions(const MutDAGGenInstructions&) = delete;
  MutDAGGenInstructions(MutDAGGenInstructions&&) = delete;

  const adt::Stack<const DAGGenInstruction>& instructions() const {
    return instructions_;
  }

  void Push(const DAGGenInstruction& instruction) {
    instructions_.push(instruction);
    UpdateNumSourceTensors(&current_num_source_tensors_, instruction);
  }

  int64_t current_num_source_tensors() const {
    return current_num_source_tensors_;
  }

 protected:
  adt::Stack<const DAGGenInstruction> instructions_;
  int64_t current_num_source_tensors_;
};

class DAGGenContext {
 public:
  explicit DAGGenContext(
      const DAGGenRequirement& requirement,
      const adt::Stack<const DAGGenInstruction>& core_dag_gen_instructions)
    : requirement_(requirement),
      core_dag_gen_instructions_(core_dag_gen_instructions) {}
  DAGGenContext(const DAGGenContext&) = delete;
  DAGGenContext(DAGGenContext&&) = delete;

  template <typename ConverterT>
  void GenerateOneInstruction(const ConverterT& Converter) {
    core_dag_gen_instructions_.TryPop();
    int64_t num_core_source_tensors =
        core_dag_gen_instructions_.current_num_source_tensors();
    int64_t num_source_tensors =
        result_dag_gen_instructions_.current_num_source_tensors();
    const DAGGenInstruction new_instruction =
        Converter(num_core_source_tensors, num_source_tensors);
    const bool is_valid = [&]{
      return IsValidSourceTensorIndex(
            new_instruction, num_core_source_tensors, num_source_tensors);
    }();
    if (is_valid) {
      result_dag_gen_instructions_.Push(new_instruction);
    }
  }

  const DAGGenRequirement& requirement() const {
    return requirement_;
  }

  const adt::Stack<const DAGGenInstruction>& result_instructions() const {
    return result_dag_gen_instructions_.instructions();
  }

 private:
  const DAGGenRequirement requirement_;
  ConstDAGGenInstructions core_dag_gen_instructions_;
  MutDAGGenInstructions result_dag_gen_instructions_;
};

class IrGenTypeGenerator {
 public:
  explicit IrGenTypeGenerator(const DAGGenTypePickProbability& pick_probability) 
    : rolling_ranges_(IrGenTypeGenerator::MakeRollingRange(pick_probability)) {}

  IrGenType<std::monostate> GetRandomIrGenType(
      int64_t num_core_source_tensors,
      int64_t num_source_tensors,
      const DAGGenRequirement& requirement) const {
    auto IsValidNumSources = [&](const auto& ir_gen_type) {
      return IsValidNumSourceTensors(
          ir_gen_type,
          num_core_source_tensors,
          num_source_tensors,
          requirement);
    };
    auto Roll = [&]()-> std::optional<IrGenType<std::monostate>> {
      size_t random_int = GetRandomInt(0, RollingLimit());
      for (const auto& [start, end, ir_gen_type] : rolling_ranges_) {
        if (!IsValidNumSources(ir_gen_type)) continue;
        if (random_int >= start && random_int < end) return ir_gen_type;
      }
      return std::nullopt;
    };
    static constexpr int kTryCnt = 10;
    for (int i = 0; i < kTryCnt; ++i) {
      if (auto type = Roll()) return type.value();
    }
    return Nope<std::monostate>{};
  }

 private:
  struct IrGenTypeRollingRange {
    size_t start;
    size_t end;
    IrGenType<std::monostate> ir_gen_type;
  };
  static std::vector<IrGenTypeRollingRange> MakeRollingRange(
      const DAGGenTypePickProbability& pick_probability) {
    const size_t total_weight = GetTotalWeight(pick_probability);
    std::vector<IrGenTypeRollingRange> ret;
    size_t start = 0;
    auto GetRange = [&](const auto& ir_gen_type) {
      size_t current_start = start;
      size_t current_end = current_start;
      float weight = GetWeight(ir_gen_type, pick_probability);
      current_end += weight * RollingLimit() / total_weight;
      start = current_end;
      return IrGenTypeRollingRange{
        .start = current_start,
        .end = current_end,
        .ir_gen_type = ir_gen_type,
      };
    };
#define PUSH_BACK_RANGE_ir_gen_type(ir_gen_type)            \
    ret.push_back(GetRange(ir_gen_type<std::monostate>{}));  \
FOR_EACH_IR_GEN_TYPE(PUSH_BACK_RANGE_ir_gen_type);
#undef PUSH_BACK_RANGE_ir_gen_type
    return ret;
  }

  static float GetTotalWeight(
      const DAGGenTypePickProbability& pick_probability) {
    float total = 0;
#define PUSH_BACK_RANGE_ir_gen_type(ir_gen_type)            \
    total += GetWeight(ir_gen_type<std::monostate>{}, pick_probability);  \
FOR_EACH_IR_GEN_TYPE(PUSH_BACK_RANGE_ir_gen_type);
#undef PUSH_BACK_RANGE_ir_gen_type
    return total;
  }

  static constexpr size_t RollingLimit() {
    return 10000;
  }

  std::vector<IrGenTypeRollingRange> rolling_ranges_;
};

class DefaultDAGGenerator final : public DAGGenerator {
 public:
  explicit DefaultDAGGenerator(const DAGGenRequirement& requirement)
    : DAGGenerator(requirement),
      ir_gen_type_generator_(requirement.pick_probability) {}

  // Instructions generating sink nodes of DAG are put on the top of stack.
  adt::Stack<const DAGGenInstruction> Generate(
      const adt::Stack<const DAGGenInstruction>& core_dag) override {
    DAGGenContext ctx(requirement_, core_dag);
    auto MakeInstruction = [&](int64_t num_core_sources, int64_t num_sources){
      return MakeRandomInstruction(ctx, num_core_sources, num_sources);
    };
    for (int64_t i = 0; i < requirement_.max_instructions; ++i) {
      ctx.GenerateOneInstruction(MakeInstruction);
    }
    return Reverse(ctx.result_instructions());
  }

 private:
  adt::Stack<const DAGGenInstruction> Reverse(
      const adt::Stack<const DAGGenInstruction>& stack) {
    adt::Stack<const DAGGenInstruction> reversed;
    for (auto s = stack; !s.empty(); s.pop()) {
      reversed.push(s.top());
    }
    return reversed;
  }

  const DAGGenInstruction MakeRandomInstruction(
      const DAGGenContext& ctx,
      int64_t num_core_source_tensors,
      int64_t num_source_tensors) {
    PADDLE_ENFORCE_GE(num_core_source_tensors, 0);
    PADDLE_ENFORCE_GE(num_core_source_tensors, 0);
    PADDLE_ENFORCE_LE(num_core_source_tensors, num_source_tensors);
    const auto& ir_gen_type =
      ir_gen_type_generator_.GetRandomIrGenType(
          num_core_source_tensors, num_source_tensors, requirement_);
    return GenerateDAGGenInstruction(
        ir_gen_type,
        requirement_,
        num_core_source_tensors,
        num_source_tensors);
  }

  const IrGenTypeGenerator ir_gen_type_generator_;
};

}

std::unique_ptr<DAGGenerator> MakeDefaultDAGGenerator(
    const DAGGenRequirement& requirement) {
  return std::make_unique<DefaultDAGGenerator>(requirement);
}

}