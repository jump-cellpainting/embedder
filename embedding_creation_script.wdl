# Copyright 2022 Verily Life Sciences LLC
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd
#
# Optimized version of the Embedding Creation workflow.
#
# Coding standard https://biowdl.github.io/styleGuidelines.html is used with newer command body
# style https://github.com/openwdl/wdl/blob/main/versions/1.0/SPEC.md#command-section.

version 1.0

import 'https://raw.githubusercontent.com/broadinstitute/cellprofiler-on-Terra/v0.2.0/utils/cellprofiler_distributed_utils.wdl' as cell_profiler_workflows

# TODO(deflaux) add more documentation and parameters_meta after this gets a review from mando@
# TODO(deflaux, mando) store the Python scripts in the Docker image instead of pulling them from GCS, after
#   we are finished changing them so frequently. That will eliminate two parameters and make this workflow easier for others to run.
# TODO(deflaux) add AWS federated auth for Google Accounts

workflow EmbeddingCreation {
    input {
            
        #--- [ sharding specific parameters ] ----
        # Python script to perform the sharding.
        File shardingScript
        # Specify path to input load_data_with_illum.csv, which contains paths to *.tiff, and illum/*.npy.
        String loadDataWithIllum
        # To determine the shard for a well, perform the modulus operation over the last two digits of the well name
        # or use the numeric value verbatim if the modulus value is zero.
        Int modulus = 0

        #--- [ embedding creation specific parameters ] ----
        # GCS path to Python script version of embedding_creation.ipynb
        File embeddingCreationScript
        # GCS or S3 path underneath which computed cell centers are stored.
        String cellCentersPathPrefix
        # Desired location of the computed embeddings.
        String embeddingOutputPath
        String sourceId
        String batchId
        String plateId
        Int cellPatchDim
        Int modelBatchDim
        String tfHubModelPath
        Int tfHubModelInputImageHeight
        Int tfHubModelInputImageWidth
        Int tfHubModelOutputEmbSize
        String embeddingCreationDockerImage = 'gcr.io/terra-solutions-jump-cp-dev/embedding_creation:20220929_151238'
        Int embeddingCreationCPU = 8
        Int embeddingCreationMemoryGB = 30
        Int embeddingCreationDiskGB = 10
        Int embeddingCreationMaxRetries = 1
        Int embeddingCreationPreemptibleAttempts = 2
        String embeddingCreationGPUType = "nvidia-tesla-t4"
        Int embeddingCreationGPUCount = 1
        String embeddingCreationNvidiaDriverVersion = "470.82.01"
        String embeddingCreationZone = 'us-central1-c'

        # Optional: If the image or cell center input files are in an S3 bucket, this workflow can read the files directly from AWS.
        # To configure this:
        # 1) Store the AWS access key id and secret access key in Google Cloud Secret Manager. This allows the secret to be used
        #    by particular people without it being visible to everyone who can see the workspace.
        #    (https://cloud.google.com/secret-manager/docs/create-secret)
        # 2) Grant permission 'Secret Manager Secret Accessor' to your personal Terra proxy group.
        #    (https://support.terra.bio/hc/en-us/articles/360031023592-Pet-service-accounts-and-proxy-groups-)
        # 3) Pass the secret's "Resource ID" as the value to these workflow parameters.
        String? secret_manager_resource_id_aws_access_key_id
        String? secret_manager_resource_id_aws_secret_access_key

    }

    String embeddingOutputPathTrimmed = sub(embeddingOutputPath, '/+$', '')

    # Determine which wells should be processed within which shards.
    call determineShards {
        input:
            shardingScript = shardingScript,
            loadDataWithIllum = loadDataWithIllum,
            modulus = modulus,
            secret_manager_resource_id_aws_access_key_id = secret_manager_resource_id_aws_access_key_id,
            secret_manager_resource_id_aws_secret_access_key = secret_manager_resource_id_aws_secret_access_key,
            dockerImage = embeddingCreationDockerImage
    }

    # Run embedding creation scattered by shards of multiple wells.
    scatter(shard in determineShards.value) {
        call runEmbeddingCreationScript {
            input:
                embeddingCreationScript = embeddingCreationScript,
                shardMetadata = shard,
                loadDataWithIllum = loadDataWithIllum,
                cellCentersPathPrefix = cellCentersPathPrefix,
                secret_manager_resource_id_aws_access_key_id = secret_manager_resource_id_aws_access_key_id,
                secret_manager_resource_id_aws_secret_access_key = secret_manager_resource_id_aws_secret_access_key,
                sourceId = sourceId,
                batchId = batchId,
                plateId = plateId,
                cellPatchDim = cellPatchDim,
                modelBatchDim = modelBatchDim,
                tfHubModelPath = tfHubModelPath,
                tfHubModelInputImageHeight = tfHubModelInputImageHeight,
                tfHubModelInputImageWidth = tfHubModelInputImageWidth,
                tfHubModelOutputEmbSize = tfHubModelOutputEmbSize,
                dockerImage = embeddingCreationDockerImage,
                cpu = embeddingCreationCPU,
                memoryGB = embeddingCreationMemoryGB,
                diskGB = embeddingCreationDiskGB,
                maxRetries = embeddingCreationMaxRetries,
                preemptibleAttempts = embeddingCreationPreemptibleAttempts,
                gpuType = embeddingCreationGPUType,
                gpuCount = embeddingCreationGPUCount,
                nvidiaDriverVersion = embeddingCreationNvidiaDriverVersion,
                zone = embeddingCreationZone
         }

        call cell_profiler_workflows.extract_and_gsutil_rsync as delocalizeEmbeddingOutputs {
            input:
                tarball=runEmbeddingCreationScript.tarOutputs,
                destination_gsurl=embeddingOutputPathTrimmed
        }
    }

    output {
        String outputDirectory = delocalizeEmbeddingOutputs.output_directory[0]
        Array[File] tarOutputs = runEmbeddingCreationScript.tarOutputs
        Array[File] dataWarningsLog = runEmbeddingCreationScript.dataWarningsLog
    }
}

task determineShards {

    input {
        # Python script to perform the sharding.
        File shardingScript
        # Specify path to input load_data_with_illum.csv, which contains GCS paths to *.tiff, and illum/*.npy.
        String loadDataWithIllum
        # To determine the shard for a well, perform the modulus operation over the last two digits of the well name
        # or use the numeric value verbatim if the modulus value is zero.
        Int modulus = 0
        # Optional: If the CellProfiler analysis results are in an S3 bucket, this workflow can read the files directly from AWS.
        # To configure this:
        # 1) Store the AWS access key id and secret access key in Google Cloud Secret Manager. This allows the secret to be used
        #    by particular people without it being visible to everyone who can see the workspace.
        #    (https://cloud.google.com/secret-manager/docs/create-secret)
        # 2) Grant permission 'Secret Manager Secret Accessor' to your personal Terra proxy group.
        #    (https://support.terra.bio/hc/en-us/articles/360031023592-Pet-service-accounts-and-proxy-groups-)
        # 3) Pass the secret's "Resource ID" as the value to these workflow parameters.
        String? secret_manager_resource_id_aws_access_key_id
        String? secret_manager_resource_id_aws_secret_access_key

        # Docker image
        String dockerImage = 'gcr.io/terra-solutions-jump-cp-dev/embedding_creation:20220929_151238'
    }

    String outputFilename = 'shards_metadata.txt'
    String outputScriptFilename = 'executed_script.py'
    
    command <<<
        # Errors should cause the task to fail, not produce an empty output.
        set -o errexit
        set -o pipefail
        set -o nounset

        ~{if defined(secret_manager_resource_id_aws_access_key_id)
          then "export AWS_ACCESS_KEY_ID=$(gcloud secrets versions access ~{secret_manager_resource_id_aws_access_key_id})"
          else ""
        }

        ~{if defined(secret_manager_resource_id_aws_secret_access_key)
          then "export AWS_SECRET_ACCESS_KEY=$(gcloud secrets versions access ~{secret_manager_resource_id_aws_secret_access_key})"
          else ""
        }

        # Send a trace of all fully resolved executed commands to stderr.
        # Note that we enable this _after_ fetching credentials, because we do not want to log those values.
        set -o xtrace
        python3 ~{shardingScript} \
            --load_data_with_illum_csv_file=~{loadDataWithIllum} \
            --modulus=~{modulus} \
            --output_filename=~{outputFilename}

        # Capture a copy of the executed notebook for the sake of provenance.
        cp ~{shardingScript} ~{outputScriptFilename}
    >>>

    output {
        Array[String] value = read_lines(outputFilename)
        File outputText = outputFilename
        File executedScript = outputScriptFilename
    }

    runtime {
        docker: dockerImage
        maxRetries: 1
        preemptible: 2
    }

}

task runEmbeddingCreationScript {
    input {        
        # GCS path to Python script version of embedding_creation.ipynb
        File embeddingCreationScript
        # GCS or S3 path underneath which computed cell centers are stored.
        String shardMetadata
        # Specify path to input load_data_with_illum.csv, which contains GCS paths to *.tiff, and illum/*.npy.
        String loadDataWithIllum
        String cellCentersPathPrefix
        # Optional: If the CellProfiler analysis results are in an S3 bucket, this workflow can read the files directly from AWS.
        # To configure this:
        # 1) Store the AWS access key id and secret access key in Google Cloud Secret Manager. This allows the secret to be used
        #    by particular people without it being visible to everyone who can see the workspace.
        #    (https://cloud.google.com/secret-manager/docs/create-secret)
        # 2) Grant permission 'Secret Manager Secret Accessor' to your personal Terra proxy group.
        #    (https://support.terra.bio/hc/en-us/articles/360031023592-Pet-service-accounts-and-proxy-groups-)
        # 3) Pass the secret's "Resource ID" as the value to these workflow parameters.
        String? secret_manager_resource_id_aws_access_key_id
        String? secret_manager_resource_id_aws_secret_access_key
        String sourceId
        String batchId
        String plateId
        Int cellPatchDim
        Int modelBatchDim
        String tfHubModelPath
        Int tfHubModelInputImageHeight
        Int tfHubModelInputImageWidth
        Int tfHubModelOutputEmbSize
        
        String dockerImage = 'gcr.io/terra-solutions-jump-cp-dev/embedding_creation:20220929_151238'
        Int cpu = 8
        Int memoryGB = 30
        Int diskGB = 10
        Int maxRetries = 1
        Int preemptibleAttempts = 2
        String gpuType = 'nvidia-tesla-t4'
        Int gpuCount = 1
        String nvidiaDriverVersion = '470.82.01'
        String zone = 'us-central1-c'

    }
  
    String workDir = 'workdir'
    String tarOutputsFile = 'outputs.tar.gz'
    String outputScriptFilename = 'executed_script.py'

    command <<<
        # Errors should cause the task to fail, not produce an empty output.
        set -o errexit
        set -o pipefail
        set -o nounset

        ~{if defined(secret_manager_resource_id_aws_access_key_id)
          then "export AWS_ACCESS_KEY_ID=$(gcloud secrets versions access ~{secret_manager_resource_id_aws_access_key_id})"
          else ""
        }

        ~{if defined(secret_manager_resource_id_aws_secret_access_key)
          then "export AWS_SECRET_ACCESS_KEY=$(gcloud secrets versions access ~{secret_manager_resource_id_aws_secret_access_key})"
          else ""
        }

        # Send a trace of all fully resolved executed commands to stderr.
        # Note that we enable this _after_ fetching credentials, because we do not want to log those values.
        set -o xtrace

        mkdir -p ~{workDir}
        cd ~{workDir}

        python3 ~{embeddingCreationScript} \
            --shard_metadata='~{shardMetadata}' \
            --cell_center_path_prefix=~{cellCentersPathPrefix} \
            --source_id=~{sourceId} \
            --batch_id=~{batchId} \
            --plate_id=~{plateId} \
            --load_data=~{loadDataWithIllum} \
            --cell_patch_dim=~{cellPatchDim} \
            --model_batch_dim=~{modelBatchDim} \
            --tf_hub_model_path=~{tfHubModelPath} \
            --tf_hub_model_output_emb_height=~{tfHubModelInputImageHeight} \
            --tf_hub_model_output_emb_width=~{tfHubModelInputImageWidth} \
            --tf_hub_model_output_emb_size=~{tfHubModelOutputEmbSize}
       
        # Create a tar to also capture any outputs written to subdirectories, in addition to the current working directory.
        cd ..
        tar -zcvf ~{tarOutputsFile} --directory ~{workDir} .
        # Capture a copy of the executed notebook for the sake of provenance.
        cp ~{embeddingCreationScript} ~{outputScriptFilename}

    >>>

    output {
        File tarOutputs = tarOutputsFile
        File executedScript = outputScriptFilename
        File dataWarningsLog = glob('*data_warnings.log')[0]
    }

    # See also https://cromwell.readthedocs.io/en/stable/RuntimeAttributes/#recognized-runtime-attributes-and-backends
    # How to configure GPUs https://support.terra.bio/hc/en-us/articles/360055066731
    runtime {
        docker: dockerImage
        memory: memoryGB + ' GB'
        disks: 'local-disk ' + diskGB + ' SSD'
        maxRetries: maxRetries
        preemptible: preemptibleAttempts
        cpu: cpu
        gpuType: gpuType
        gpuCount: gpuCount
        nvidiaDriverVersion: nvidiaDriverVersion
        zones: [zone]
    }

}
