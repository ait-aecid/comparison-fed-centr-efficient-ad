class Color:
    @staticmethod
    def purple(msg: str) -> str:
        return f"\033[95m{msg}\033[0m"

    @staticmethod
    def blue(msg: str) -> str:
        return f"\033[94m{msg}\033[0m"

    @staticmethod
    def green(msg: str) -> str:
        return f"\033[92m{msg}\033[0m"

    @staticmethod
    def yellow(msg: str) -> str:
        return f"\033[93m{msg}\033[0m"

    @staticmethod
    def red(msg: str) -> str:
        return f"\033[91m{msg}\033[0m"