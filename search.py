import os
from hyperopt import fmin, tpe, hp, Trials
import random

def fun_hp(param):
    alpha = param['alpha'] * 1.0 / 100000
    temperature = param['temperature'] * 1.0 / 20
    print(alpha,temperature)
    avg_metrics_maf1 = 0
    avg_metrics_mif1 = 0
    avg_metrics_js = 0
    for seed in [1111, 2222, 3333, 4444, 5555]:
        os.system(
            'python scripts/train.py --train-path data/E-c-In-train.txt --dev-path data/E-c-In-dev.txt --loss-type SCL --seed %f --lang Indonesian --alpha-loss %f --temperature %f' % (seed, alpha, temperature)
        )

        os.system(
            'python scripts/test.py --test-path data/E-c-In-test.txt --model-path %s_checkpoint.pt --lang Indonesian' % str(seed)
        )

        with open('result.txt') as f:
            metrics = f.read().strip().split('\t')
            avg_metrics_maf1 += float(metrics[0])
            avg_metrics_mif1 += float(metrics[1])
            avg_metrics_js += float(metrics[2])

    avg_metrics_maf1 = avg_metrics_maf1 / 5
    avg_metrics_mif1 = avg_metrics_mif1 / 5
    avg_metrics_js = avg_metrics_js / 5
    save_data = [str(alpha), str(temperature), str(avg_metrics_maf1), str(avg_metrics_mif1), str(avg_metrics_js)]
    g = open('trial.txt', 'a')
    g.write('\t'.join(save_data)+'\n')
    g.flush()
    g.close()
    return -float(avg_metrics_maf1)-float(avg_metrics_mif1)-float(avg_metrics_js)

# JSPCL
# space_hp = {
#     'alpha': hp.uniformint('alpha',1,100),
#     'temperature': hp.uniformint('temperature',14,40)
# }

# JSCL
# space_hp = {
#     'alpha': hp.uniformint('alpha',1,5000),
#     'temperature': hp.uniformint('temperature',10,40)
# }

# SLCL
# space_hp = {
#     'alpha': hp.uniformint('alpha',1,10000),
#     'temperature': hp.uniformint('temperature',10,40)
# }

# CSL
# space_hp = {
#     'alpha': hp.uniformint('alpha',1,1000),
#     'temperature': hp.uniformint('temperature',10,40)
# }

# SCL
space_hp = {
    'alpha': hp.uniformint('alpha',1,10000),
    'temperature': hp.uniformint('temperature',10,40)
}


trials = Trials()
param = fmin(fun_hp, space_hp, tpe.suggest, max_evals=100, trials=trials)
print(param)