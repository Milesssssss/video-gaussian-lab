"""Phase 1 执行路径上的可区分失败。"""


class ContractError(ValueError):
    """输入违反数据契约或实验 ID 规则。"""


class OfficialRunRefused(RuntimeError):
    """正式协议运行被拒绝。拒绝不是算法失败，也不是已测得的质量结果。"""


class LeakageError(ContractError):
    """留出帧进入了重建上下文。"""


class Dp2IsolationError(ContractError):
    """SHARP-only 对照读取了 Depth Pro 2 输出。"""


class ModelPreconditionError(RuntimeError):
    """模型权重、许可或运行环境不满足，推理未执行。"""


class IndependentReferenceError(ContractError):
    """把模型自身预测当成了独立几何真值。"""


class RendererNotAvailable(RuntimeError):
    """没有已验证的共享场景渲染器，优化步骤未执行。"""


class ScaleNotApplied(ContractError):
    """尺度估计失败，调用方不得改写高斯。"""
