# embedder

> [!CAUTION]
> This repository is no longer being used as of Aug 4, 2025. All issues were marked as "Not planned" and added to this [project](https://github.com/orgs/jump-cellpainting/projects/16/views/1).

Method to compute per-cell embeddings for all JUMP-CP data.

## How to use this method

In the instructions below we use this method via [Docstore](https://dockstore.org/) and [Terra](https://app.terra.bio/) to generate embeddings for one plate from the `cpg0016-jump` [dataset](https://github.com/jump-cellpainting/datasets). 
When tested in May 2024, this example plate took less than 1 hour to process via the workflow and cost $3.

1. Download  to your local file system the file [single_plate_example_inputs.json](single_plate_example_inputs.json) which holds input parameters for an example JUMP-CP plate.
2. Go to [https://dockstore.org/workflows/github.com/jump-cellpainting/embedder](https://dockstore.org/workflows/github.com/jump-cellpainting/embedder:main?tab=info) and click on `Launch with: Terra`.
3. Either create a new [Terra](https://app.terra.bio/) workspace or choose an existing workspace into which to import the workflow method. Either way, you will be automatically redirected to the Terra page for the newly imported workflow.
4. On the [Terra](https://app.terra.bio/) workflow page:

    1. Select `Run workflow with inputs defined by file paths`.
    2. Then click on `Drag or click to upload json` and choose the file `single_plate_example_inputs.json` which you downloaded in the first step.
    3. Set the value of parameter `embeddingOutputPath` to be a Google Cloud Storage path of your choosing, such as a folder within your [Terra Workspace bucket](https://support.terra.bio/hc/en-us/articles/360026059391-Where-s-the-link-for-a-file-in-workspace-storage).
    4. Click `Save`
    5. Click `Run Analysis`

When the workflow completes, Terra will send a notification email.
