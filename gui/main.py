import sys

from gui.app import create_app, create_main_window


def main() -> int:
    app = create_app()
    window = create_main_window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
