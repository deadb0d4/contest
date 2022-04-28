def get_formatter(pat: str):
    return lambda s, *args: pat.format(*args)

def build_localizer(keys: dict):
    fields = dict()
    for k, v in keys.items():
        fields[k] = get_formatter(v)
    return type('Localizer', (), fields)
