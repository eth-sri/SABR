Usage:

``` bash
source setup_on_euler.sh
bsub -n 1 -R "rusage[mem=40960,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAA100_PCIE_40GB]" -W 24:00 "python UT1.py --batch-size 256 --accum-batches 8 --use-cauchy --save-freq 50 --epochs 1500 --custom-schedule '[1000, 1400]' --start-epoch-eps 20 --end-epoch-eps 200 --start-epoch-kappa 20 --end-epoch-kappa 20 --lr 0.5"
```
