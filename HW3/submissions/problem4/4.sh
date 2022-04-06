python main.py --dataset cifar10 --model resnet --model-config "{'depth': 44}"                           -b 64 --epochs 100 --save resnet44_simple_aug
python main.py --dataset cifar10 --model resnet --model-config "{'depth': 44}"  --duplicates 2 --cutout -b 64 --epochs 100 --save resnet44_cutout_m-2
python main.py --dataset cifar10 --model resnet --model-config "{'depth': 44}"  --duplicates 4 --cutout -b 64 --epochs 100 --save resnet44_cutout_m-4
python main.py --dataset cifar10 --model resnet --model-config "{'depth': 44}"  --duplicates 8 --cutout -b 64 --epochs 100 --save resnet44_cutout_m-8
python main.py --dataset cifar10 --model resnet --model-config "{'depth': 44}"  --duplicates 16 --cutout -b 64 --epochs 100 --save resnet44_cutout_m-16
python main.py --dataset cifar10 --model resnet --model-config "{'depth': 44}"  --duplicates 32 --cutout -b 64 --epochs 100 --save resnet44_cutout_m-32