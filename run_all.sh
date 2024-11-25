echo "##############################################################################"
echo "[Main] Start - experiment"
echo

echo "##############################################################################"
echo "[Main] Experiment: 0_synthetic_exp.py"
python 0_synthetic_exp.py --dim 100 --n-samples 500 --m-max 50 --l-lbda 0.01 0.1 1.0 --cov-mat-type 'toeplitz' --n-trials 10 --n-jobs 5
echo "===================================================="
python 0_synthetic_exp.py --dim 200 --n-samples 200 --m-max 50 --l-lbda 0.01 0.1 1.0 --cov-mat-type 'identity' --n-trials 10 --n-jobs 5
echo "===================================================="

echo "##############################################################################"
echo "[Main] Experiment: 1_real_data_exp.py"
python3.10 1_real_data_exp.py --l-lbda 0.01 0.1 1.0 --n-trials 10 --n-jobs 5
echo "===================================================="
echo

echo "##############################################################################"
echo "[Main] - experiment done"

rm -rf __cache__/ __pycache__/
