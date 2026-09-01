# TODO

## Anas (A)
- [ ] Check Layer toggling code
- [ ] Verify Merge of layers on central server
- [ ] Fix sharing of attributes
- [ ] Fix default model loading
- [ ] Move dataset generation to nodes (A-E)
- [ ] Verify test/val dataset partition

## Ehtisham (E)
- [ ] Add non-iid partitioner
  - In `melocare/yolosimulation/data_partition.py`, add a function that splits dataset more aggressively
  - Select parameters that give full control on the function
  - Refer to research papers on what kind of data split is enough for non-iid split
  - Do not just make up a split algorithm

- [ ] Write script to verify - against central - against MMDB, compare and save results
  - In `myelocare/central/test` there are .ipynb files that test .pt file on real data
  - Formulate workflow to do the same for all final central models in yolosimulation runs and central runs
  - Check `myelocare/yolosimulation/server_app.py` main function. Here `final_model.pt` is saved
  - Maybe we can call all the analysis including graph plots here and save them in runs folder
  - Suggested path: `os.environ["RUN_DIR"]/server_aggregated_results`

- [ ] Add code for graph plots
  - Keep scripts separate, maybe in `scripts/plot` folder
  - Use in above task or wherever you see fit

- [ ] Move dataset generation to nodes (A-E)
  - Synthetic dataset generation should be on each node
  - After everything else is completed

## Notes
- **A** = Anas
- **E** = Ehtisham
- Items with "A - E" indicate shared/collaborative tasks
- Make new git feature branch for each task
- Don't push unnecessary code changes
- We are using flwr version 1.25.0 - make sure to refer to that version's docs to avoid incompatibility issues. Newer versions have very different APIs