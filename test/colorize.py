def colorize(x, cs):
    palette = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "bold": "\033[1m",
        "underline": "\033[4m",
        "endc": "\033[0m",
    }
    for c in cs:
        x = palette[c] + x + palette["endc"]
    return x


def err_style(x):
    return colorize(x, ["red", "bold"])
