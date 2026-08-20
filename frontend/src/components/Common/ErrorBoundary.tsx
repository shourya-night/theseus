import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("THESEUS UI Caught Error:", error, errorInfo);
    this.setState({ errorInfo });
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="w-screen h-screen flex flex-col items-center justify-center bg-[#04060a] text-[#e6dfd5] font-mono p-6 select-text">
          <div className="max-w-xl w-full bg-[#070d18] border border-[#cc3333] p-6 rounded space-y-4 shadow-2xl">
            <div className="flex items-center space-x-2 text-[#cc3333] font-bold text-sm">
              <AlertTriangle className="w-5 h-5" />
              <span>THESEUS / UI RUNTIME EXCEPTION CAUGHT</span>
            </div>

            <p className="text-xs text-[#c8c0b5] leading-relaxed">
              The application encountered a component rendering exception. The engine has contained the error to prevent application failure.
            </p>

            <div className="bg-[#05080f] border border-[#221d17] p-3 rounded text-[11px] text-[#ffaa22] overflow-x-auto max-h-40">
              <div className="font-bold mb-1">ERROR MESSAGE:</div>
              <pre className="whitespace-pre-wrap">{this.state.error?.message || "Unknown error"}</pre>
            </div>

            {this.state.error?.stack && (
              <div className="bg-[#05080f] border border-[#221d17] p-3 rounded text-[10px] text-[#8c8275] overflow-x-auto max-h-32">
                <div className="font-bold mb-1 text-[#8c8275]">STACK TRACE:</div>
                <pre className="whitespace-pre-wrap">{this.state.error.stack}</pre>
              </div>
            )}

            <button
              onClick={this.handleReset}
              className="w-full bg-[#ff9900] hover:bg-[#ffaa22] text-[#04060a] font-bold py-2.5 px-4 rounded text-xs flex items-center justify-center space-x-2 transition-all cursor-pointer"
            >
              <RotateCcw className="w-4 h-4" />
              <span>RELOAD MISSION ENVIRONMENT</span>
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
