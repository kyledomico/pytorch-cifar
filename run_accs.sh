# !/bin/bash 

# Run the accuracy test for the given model
python3 check_acc.py --model resnet34
python3 check_acc.py --model resnet50
python3 check_acc.py --model resnet101
python3 check_acc.py --model resnet152