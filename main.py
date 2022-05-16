from code import hyper_parameter_search as hps
import fire

def main(index=0, num_repetition=3, run_name=''):
    model_ckpt_base = f'/home/scl1pal/projects/NeuralLVM/exp/{run_name}_index_{index}'
    config = hps.get_config(index, num_repetition=num_repetition, num_seeds=1000000, global_seed=42)  # global_seed immer schoen gleich lassen, sonst machst du den seed liste kaputt
    print('index', index)
    print('config', config)
    for r in range(num_repetition):
        hps.run_experiment(config, repetition=r, model_ckpt_base=model_ckpt_base)


if __name__ == '__main__':
    fire.Fire()