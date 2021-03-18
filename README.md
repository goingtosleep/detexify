# Detexifier

Tex symbol classifier similar to [Detexify](https://detexify.kirelabs.org/) but using CNN.

## Data
Input dataset can be found [here](https://drive.google.com/file/d/1PNA95QKiyhWkfntP4BtpEsDVvktY0Fr7/view?usp=sharing).

## Results
Accuracy in form `train - test`.
Train | valid | test = 80% | 10% | 10%.

Simple CNN
`[Acc: (top1) 73.52 75.77 | (top5) 97.01 98.16]`

Resnet18
`[Acc: (top1) 89.25 72.64 | (top5) 99.66 97.27]`

CNN with shortcut (resnet-like) and dropblock
`[Acc: (top1) 83.88 74.79 | (top5) 99.34 98.11]`

Above but stratified data
`[Acc: (top1) 84.92 74.53 | (top5) 99.42 98.38]`
