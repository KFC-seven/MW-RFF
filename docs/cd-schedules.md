# Choosing the frame order in XFR-CD

Supply one ordered pool per domain for the same transmitter. With `D` domains and block size `m`, each domain supplies `m/D` references. The order of the pools fixes the domain order. Split the physical references into the intended data roles before constructing blocks for a new study.

| Strategy in `cd_blocks` | Example with domains A, B, C and 12 references | Paper example |
|---|---|---|
| `round_robin` | A B C · A B C · A B C · A B C | The WiSig 20 dB model in Table V used one reference per receiver per visit. |
| `short_runs`, `chunk_size=2` | AA BB CC · AA BB CC | WiSig used groups of five per receiver in the cross-date and multi-frame profiles. The receiver-disjoint evaluation also interleaved five-reference groups from its four receivers. |
| `domain_groups` | AAAA BBBB CCCC | The LTE-V Table VIII profile takes 32 consecutive references from each of eight receiver/speed domains (`m=256`). |

Each policy draws the next unused references from its domain pool. Pool order can be recording order or a prescribed permutation, as in the WiSig multi-frame profile. The final XFR transpose takes place after this scheduling step. A row at reference position `n` retains position `n` from every frame in the scheduled order.

Use SD when the input consists of consecutive frames from one recording. For CD, select a schedule at training time and use the same rule at inference. Changing the schedule of a pretrained model defines a different test condition. The implementation returns full balanced blocks and drops unused tails.

```shell
python examples/cd_schedules.py
```
