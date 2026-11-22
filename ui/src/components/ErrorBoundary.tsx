import { Component, type ErrorInfo, type ReactNode } from 'react';

export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error: Error, details: ErrorInfo) { console.error('工作区加载失败', error, details.componentStack); }
  render() {
    if (this.state.failed) return <div className="view-loading" role="alert"><h2 className="text-lg text-slate-100">工作区暂时无法显示</h2><p className="text-slate-400">可以切换其他页面，或重新加载当前工作区。</p><button type="button" className="secondary-button" onClick={() => this.setState({ failed: false })}>重新加载</button></div>;
    return this.props.children;
  }
}
