import registration from '../../../configs/phase1_protocol.json';

export const phase1Protocol = registration;
export type ProtocolRegistration = Omit<typeof registration, 'results'> & { results: unknown };

const documentBase = 'https://github.com/Milesssssss/video-gaussian-lab/blob/main/';
export const researchDocuments = {
  protocol: documentBase + 'docs/EXPERIMENT_PROTOCOL.md',
  registration: documentBase + 'configs/phase1_protocol.json',
  dataContract: documentBase + 'docs/DATA_CONTRACT.md',
  implementation: documentBase + 'docs/IMPLEMENTATION_STATUS.md',
  thirdParty: documentBase + 'docs/THIRD_PARTY.md',
  backlog: documentBase + 'docs/PHASE1_BACKLOG.md',
};

const conditionDescriptions = {
  data_available: { label: '真实静态视频与场景清单', detail: '开发 / 测试场景、文件哈希和精确帧 ID。' },
  camera_contract_verified: { label: '相机与时间同步审计', detail: '标定 K、固定米制位姿、图像变换和同步记录。' },
  independent_geometry_available: { label: '独立几何参考', detail: '协议允许 G0 = NA；当前 runner 守卫仍要求此项。' },
  model_use_conditions_resolved: { label: '模型与数据使用条件', detail: '代码、权重和采集数据分别登记适用范围。' },
  hardware_profiled: { label: '运行硬件与资源测量', detail: '实际硬件、完整耗时及峰值显存。' },
  runtime_versions_pinned: { label: '运行环境版本', detail: '依赖、模型源码和执行器版本。' },
  weight_checksums_recorded: { label: '权重校验和', detail: '适用使用条件确认后记录权重 SHA-256。' },
} satisfies Record<keyof ProtocolRegistration['entry_conditions'], { label: string; detail: string }>;

/** Registration facts only. This does not probe a worker or authorize an experiment. */
export function getResearchStatus(protocol: ProtocolRegistration = phase1Protocol) {
  const conditions = (Object.keys(conditionDescriptions) as Array<keyof typeof conditionDescriptions>)
    .map((key) => ({
      key,
      ...conditionDescriptions[key],
      met: protocol.entry_conditions[key] === true,
      requiredByProtocol: key !== 'independent_geometry_available',
      requiredByCurrentGuard: true,
    }));

  return {
    conditions,
    unmetExecutionConditions: conditions.filter((condition) => condition.requiredByCurrentGuard && !condition.met),
    unmetProtocolConditions: conditions.filter((condition) => condition.requiredByProtocol && !condition.met),
    executionEnabled: protocol.execution_enabled === true,
    implementationAvailable: protocol.implementation_exists === true,
    geometryState: protocol.entry_conditions.independent_geometry_available ? 'reference_registered' : 'NA',
    resultState: protocol.results == null ? 'not_provided' : 'payload_not_connected',
    protocolState: protocol.status,
  } as const;
}

export const armDescriptions: Record<string, string> = {
  S0: '无优化诊断',
  S1: '共享 RGB 优化基线',
  S2: '每关键帧单标量尺度初始化',
  S3: '增加可见表面深度残差',
  D0: '普通深度高斯初始化',
  D1: '普通高斯 + 深度残差',
};
