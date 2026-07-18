"""Application entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="课程资料问答智能体")
    parser.add_argument("--screenshot", help="以演示模式生成界面截图后退出")
    args = parser.parse_args()
    if args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication

        from ui import MainWindow

        app = QApplication(sys.argv)
        window = MainWindow(demo=True)
        window.show()
        output = Path(args.screenshot).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        def capture() -> None:
            window.repaint()
            app.processEvents()
            window.grab().save(str(output), "PNG")
            app.quit()

        QTimer.singleShot(1600, capture)
        return app.exec_()
    from ui import run_app

    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
