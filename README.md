From Human Labels to Literature: Semi-Supervised Learning of NMR Chemical Shifts at Scale
==================================================

![framework](./figure/framework.jpg)

## Environment
```bash
pip install -r requirements.txt
```

## Datasets & Models
All pretrained weights, datasets, and trained checkpoints are available on [Zenodo](https://zenodo.org/records/18232165).
- Download the pretrained weights and place them in `./weight/`.
- Download the datasets and place them in `./data/`.


## Running
- **H/C**  
  ```bash
  sh script/run_H.sh   # or run_C.sh
  ```
- **H/C with solvent**  
  ```bash
  sh script/run_H_with_solv.sh
  ```
- **Heteroatoms (F / P / B / Si)**  
  ```bash
  sh script/run_heteroatoms.sh F   # or P/B/Si
  ```

## License
This project is licensed under the terms of the MIT license. See [LICENSE](./LICENSE) for additional details.
