# embedder
Method to compute per-cell embeddings for all JUMP-CP data.

## How to use this method

In the instructions below we use this method via [Docstore](https://dockstore.org/) and [Terra](https://app.terra.bio/) to generate embeddings for one JUMP-CP plate. When tested in May of 2024, this example plate took less than 1 hour to process via the workflow and cost $3.

#. Download  to your local file system the file [single_plate_example_inputs.json](https://github.com/jump-cellpainting/embedder/blob/main/single_plate_example_inputs.json) which holds input parameters for an example JUMP-CP plate.
#. Go to [https://dockstore.org/workflows/github.com/jump-cellpainting/embedder](https://dockstore.org/workflows/github.com/jump-cellpainting/embedder:main?tab=info) and click on `Launch with: Terra`.
#. Either create a new [Terra](https://app.terra.bio/) workspace or choose an existing workspace into which to import the workflow method. Either way, you will be automatically redirected to the Terra page for the newly imported workflow.
#. On the [Terra](https://app.terra.bio/) workflow page:
    #. Select `Run workflow with inputs defined by file paths`.
    #. Then click on `Drag or click to upload json` and choose the file `single_plate_example_inputs.json` which you downloaded in the first step.
    #. Set the value of parameter `embeddingOutputPath` to be a Google Cloud Storage path of your choosing, such as a folder within your [Terra Workspace bucket](https://support.terra.bio/hc/en-us/articles/360026059391-Where-s-the-link-for-a-file-in-workspace-storage).
    #. Click `Save`
    #. Click `Run Analysis`
#. When the workflow completes, Terra will send a notification email.

