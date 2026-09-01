# TODO

## Anas (A)
- [ ] Check Layer toggling code
- [ ] Verify Merge of layers on central server
- [ ] Fix sharing of attributes
- [ ] Fix default model loading
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

- [ ] Move dataset generation to nodes 
  - Synthetic dataset generation should be on each node
  - [Client APP doc 1.25.0](https://flower.ai/docs/framework/1.25/en/ref-api/flwr.clientapp.ClientApp.html): There is a lifespan decorator that registers a funtion which will run before a client app. We can put the dataset generation code here. 
  - Make sure the code is configureable - I mean maximum control such as the number of new images generated etc. 
  We can share those configurations from server side in ConfigRecord - There is a problem where clients are not recieving the vars in ConfigRecord. I will debug it. For now i am using os.environ temporaririly. You can just create hardcoded Config Vars in lifespan function on top. I will migrate them to ConfigRecord Later

## Notes
- **A** = Anas
- **E** = Ehtisham
- Right now there is a problem
- Make new git feature branch for each task
- Don't push unnecessary code changes
- We are using flwr version 1.25.0 - make sure to refer to that version's docs to avoid incompatibility issues. Newer versions have very different APIs