# Attention Replace


Run main.py to train or validate models.

# There are several difference between my version and yuxian.
# 0 import 
remove numpy, torchsummary
# 1 : main.py
--data-path ----> "./data"
output_dir = "/home/xfei22/" + \
            current_time.strftime("%Y-%m-%d-%H-%M") + "/"
# 2 : datasets.py

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform,download=True)