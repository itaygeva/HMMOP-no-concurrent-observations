def get_value_or_default(var_name, config, default):
    if var_name in config.keys():
        return config[var_name]
    elif var_name in default.keys():
        return default[var_name]
    else:
        raise NotImplementedError(f"No default value set for {var_name}")

