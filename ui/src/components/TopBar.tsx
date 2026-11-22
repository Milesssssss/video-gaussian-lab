import { Box, PanelLeft, Upload } from 'lucide-react';
import type { ActiveTab } from '../types/pipeline';

interface Props {
  activeTab: ActiveTab;
  controlsOpen: boolean;
  hasVideoPreview: boolean;
  onSelectTab: (tab: ActiveTab) => void;
  onToggleControls: () => void;
  onOpenVideoModal: () => void;
}

const NAV_ITEMS: { id: ActiveTab; label: string }[] = [
  { id: 'viewport', label: '三维工作区' },
  { id: 'inspector', label: '特征与深度' },
  { id: 'benchmarks', label: '实验对照' },
  { id: 'blueprint', label: '管线方案' },
];

export function TopBar({ activeTab, controlsOpen, hasVideoPreview, onSelectTab, onToggleControls, onOpenVideoModal }: Props) {
  return (
    <header className="top-bar">
      <button type="button" className="wordmark" onClick={() => onSelectTab('viewport')} aria-label="GaussianFlow 三维工作区">
        <Box size={21} strokeWidth={1.6} aria-hidden="true" className="text-cyan-300" />
        <span>GaussianFlow<span className="wordmark-sub">Research Lab</span></span>
      </button>
      <nav aria-label="主工作区导航" className="workspace-nav">
        {NAV_ITEMS.map((item) => (
          <button key={item.id} type="button" onClick={() => onSelectTab(item.id)} aria-current={activeTab === item.id ? 'page' : undefined}
            className={`nav-item ${activeTab === item.id ? 'is-active' : ''}`}>{item.label}</button>
        ))}
      </nav>
      <div className="header-actions">
        {activeTab === 'viewport' && <button type="button" className="icon-button mobile-controls" aria-label="切换参数面板" aria-expanded={controlsOpen} aria-controls="scene-controls" onClick={onToggleControls}><PanelLeft size={18} aria-hidden="true" /></button>}
        <button type="button" className="primary-button" onClick={onOpenVideoModal}><Upload size={15} aria-hidden="true" /><span>{hasVideoPreview ? '查看视频' : '导入视频'}</span></button>
      </div>
    </header>
  );
}
