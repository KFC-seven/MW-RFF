# Data and evaluation

Run these commands from the repository root. Place each test NPZ in `data/` and the corresponding model ZIP in `weights/`. Each command uses all the checkpoints for its experiment by default. Use `--device cpu` for CPU inference or `--output results.json` to save scores. [Release assets](https://github.com/KFC-seven/MW-RFF/releases) · [Model archive list](../weights/README.md) · [Test set checksums](data-sha256.txt)

## WiSig

The four WiSig evaluations share `wisig-test.npz`. It contains the non-equalized I/Q references used from the ManySig collection, stored as float32 arrays. The commands also accept the original `ManySig.pkl`.

```shell
python evaluate.py wisig-20db --data data/wisig-test.npz
python evaluate.py wisig-date --data data/wisig-test.npz --all-pairs
python evaluate.py wisig-unseen --data data/wisig-test.npz
python evaluate.py wisig-multiframe --data data/wisig-test.npz
```

`--all-pairs` evaluates all 16 train/test date pairs, including the four same-date pairs. To evaluate one pair, replace it with, for example, `--source-date 2021-03-15 --target-date 2021-03-01`. The receiver-disjoint command evaluates the five assignments in Table IV. The multi-frame command reports block accuracy. The other three commands report row accuracy.

The WiSig bundle is adapted from *WiSig* by Samer Hanna, Sudheesha Karunaratne, and Danijela Cabric and is distributed under CC BY-NC-SA 4.0. Please follow the authors’ citation instructions for WiSig and the ORBIT testbed. [WiSig paper](https://doi.org/10.1109/ACCESS.2022.3154790) · [Original dataset and citations](https://cores.ee.ucla.edu/downloads/datasets/wisig/) · [Data license](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## LTE-V

Choose the test bundle for the model you want to run.

```shell
python evaluate.py ltev-sd-20db --data data/ltev-sd-20db-test.npz
python evaluate.py ltev-cd-20db --data data/ltev-cd-20db-test.npz
python evaluate.py ltev-multiframe --data data/ltev-multiframe-test.npz
```

The two 20 dB commands report row accuracy. The multi-frame command reports block accuracy. Each NPZ contains the DMRS references used to construct the test blocks. XFR-SD and multi-frame XFR-CD use 256-sample references. The XFR-CD 20 dB release uses 288-sample references.

You can also pass a directory of MAT recordings to `--data`. For XFR-SD use the variable-length recordings from the 256-point extraction. For multi-frame XFR-CD use the 256-point extraction with 2,999 references per recording. For the 256-point CD configuration described in the paper, use that same extraction with `ltev-cd-20db --profile paper256`. [LTE-V USRP RFFI dataset](https://ieee-dataport.org/documents/lte-v-usrp-rffi-dataset)
