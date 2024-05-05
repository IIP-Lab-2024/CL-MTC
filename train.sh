python scripts/train.py --train-path data/E-c-In-train.txt --dev-path data/E-c-In-dev.txt --loss-type SCL --seed 1111 --lang Indonesian --alpha-loss 0.00008 --temperature 1.9
python scripts/test.py --test-path data/E-c-In-test.txt --model-path 1111_checkpoint.pt --lang Indonesian
python scripts/train.py --train-path data/E-c-In-train.txt --dev-path data/E-c-In-dev.txt --loss-type SCL --seed 2222 --lang Indonesian --alpha-loss 0.00008 --temperature 1.9
python scripts/test.py --test-path data/E-c-In-test.txt --model-path 2222_checkpoint.pt --lang Indonesian
python scripts/train.py --train-path data/E-c-In-train.txt --dev-path data/E-c-In-dev.txt --loss-type SCL --seed 3333 --lang Indonesian --alpha-loss 0.00008 --temperature 1.9
python scripts/test.py --test-path data/E-c-In-test.txt --model-path 3333_checkpoint.pt --lang Indonesian
python scripts/train.py --train-path data/E-c-In-train.txt --dev-path data/E-c-In-dev.txt --loss-type SCL --seed 4444 --lang Indonesian --alpha-loss 0.00008 --temperature 1.9
python scripts/test.py --test-path data/E-c-In-test.txt --model-path 4444_checkpoint.pt --lang Indonesian
python scripts/train.py --train-path data/E-c-In-train.txt --dev-path data/E-c-In-dev.txt --loss-type SCL --seed 5555 --lang Indonesian --alpha-loss 0.00008 --temperature 1.9
python scripts/test.py --test-path data/E-c-In-test.txt --model-path 5555_checkpoint.pt --lang Indonesian
