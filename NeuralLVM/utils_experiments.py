from datetime import datetime


def get_exp_name(args_old, args_new, ignoring=("writer")):
    """
    Returns a convenient experiment name for tensorboard that compares
    arguments given to argparse to the default settings. It then
    writes the arguments where the values differ from the
    default settings into the experiment name.
    """

    for key, val in args_new.items():
        if val == "false" or val == "False":
            args_new[key] = False
        if val == "true" or val == "True":
            args_new[key] = True

    exp_name = args_new["name"] + "_"
    for key in args_old:
        old_val = args_old[key]
        if old_val != args_new[key]:
            if key in ignoring:
                continue
            val = args_new[key]
            if isinstance(val, float):
                exp_name += f"{key[:15]}{val:.3f}-"
            elif isinstance(val, str):
                exp_name += f"{key[:15]}" + val[:5] + "-"
            else:
                exp_name += f"{key[:15]}" + str(val) + "-"

    return exp_name + f'--{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'


class Nop(object):
    def nop(*args, **kw): pass
    def __getattr__(self, _): return self.nop