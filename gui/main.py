import sys

from gui.app import create_app, create_main_window


def main() -> int:
    """Launch the GUI application and return its exit code."""
    app = create_app()
    window = create_main_window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
