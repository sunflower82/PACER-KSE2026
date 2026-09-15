# Dataset layout (not redistributed)

PACER consumes the public MMHCL Clothing and Sports 5-core splits
and frozen modality features **unmodified**. This repository does
**not** ship those files.

Official release (Hugging Face):

- Clothing: https://huggingface.co/datasets/Xu-SII-BNU/MMHCL/tree/main/Clothing
- Sports: https://huggingface.co/datasets/Xu-SII-BNU/MMHCL/tree/main/Sports

The same payloads are also listed by MMHCL on
[Google Drive](https://drive.google.com/drive/folders/1yitfcangRzsWtYM1MokMyWPGPr8heBxB?usp=drive_link)
and
[Baidu Cloud](https://pan.baidu.com/s/1ZOE7BqSyrqD3rB2B2XLMcQ?pwd=pypk).
Cite [Guo et al., MMHCL](https://github.com/Xu107/MMHCL) if you use them.

## What must match the paper

| Field | Clothing | Sports |
| --- | ---: | ---: |
| Items | 23,033 | 18,357 |
| Split | per-user 8:1:1 chronological | same |
| Visual features | frozen ResNet-50 | frozen ResNet-50 |
| Text features | frozen Sentence-BERT | frozen Sentence-BERT |

Do not re-filter, re-split, or re-extract features.

## Expected layout

Place the Hugging Face trees under `data/` so the loader in
`src/utility/load_data.py` sees:

```text
data/
  Clothing/
    5-core/
      train.json
      val.json
      test.json
    image_feat.npy
    text_feat.npy
  Sports/
    5-core/
      train.json
      val.json
      test.json
    image_feat.npy
    text_feat.npy
```

JSON splits live in `5-core/`; frozen features sit next to that
directory, not inside it. The trainer resolves
`os.path.join(data_path, dataset, f"{core}-core")` for the splits
and `os.path.join(data_path, dataset, "{image,text}_feat.npy")`
for the features (`--data_path ./data`, `--dataset Clothing|Sports`,
`--core 5`).

## SHA-256 checksums

JSON hashes were computed from the Hugging Face `resolve/main`
bytes. Feature hashes are the Git LFS SHA-256 OIDs published by
the MMHCL Hugging Face tree.

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `Clothing/5-core/train.json` | 1,707,961 | `1f8fa7255caf1693bd854b6c7cc42ad4635465e7d7178cf551f11e9e4d1de842` |
| `Clothing/5-core/val.json` | 690,344 | `ff9f56431a7bb1782e2441f2968c7183066e7e61587acc9dd041524bd17f1760` |
| `Clothing/5-core/test.json` | 683,535 | `659f0847e124e4bab6935a4deeddb0f228ad23bdfb4c5ca71369893813dbf46e` |
| `Clothing/image_feat.npy` | 754,745,472 | `4a79519532715f7903ab8eecf0a2914d9832214c46209dd91565a78f1dcdd5d2` |
| `Clothing/text_feat.npy` | 94,343,296 | `c8b8c4538cb895b34717350f6e360c89748de7ddaad2b49690028409958360e2` |
| `Sports/5-core/train.json` | 1,773,973 | `c54cac0b6cffabf469ca867f0bc18b594c0edc6902b442e3a4274d7b31f52d78` |
| `Sports/5-core/val.json` | 636,052 | `b998dbbe47201fc7d087bce62b849308a9c02d419d03f6cbb29335fd8462eff5` |
| `Sports/5-core/test.json` | 622,157 | `e68bd6671c9b388af6ee585e7d45386a96023beb2e5b6f3e6ff3a53859880b85` |
| `Sports/image_feat.npy` | 601,522,304 | `36eb21fcb732742a4f5153815b4467dc5e2b0a4648207175f1ff8460d85c2669` |
| `Sports/text_feat.npy` | 75,190,400 | `8ab6e6ceb561d3f2a6ce0e2a9d003ac4f679fdce7908663528aeab689919ef2e` |

Verify locally with `sha256sum <file>`.
