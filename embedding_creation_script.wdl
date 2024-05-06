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

workflow EmbeddingCreation {
    input {

        #--- [ sharding specific parameters ] ----
        # Specify path to input load_data_with_illum.csv, which contains paths to *.tiff, and illum/*.npy.
        String loadDataWithIllum
        # To determine the shard for a well, perform the modulus operation over the last two digits of the well name
        # or use the numeric value verbatim if the modulus value is zero.
        Int modulus = 24

        #--- [ embedding creation specific parameters ] ----
        # GCS or S3 path underneath which computed cell centers are stored.
        String cellCentersPathPrefix
        # Desired location of the computed embeddings.
        String embeddingOutputPath
        Int cellPatchDim
        Int modelBatchDim
        String tfHubModelPath
        Int tfHubModelInputImageHeight
        Int tfHubModelInputImageWidth
        Int tfHubModelOutputEmbSize
        String embeddingCreationDockerImage = 'PUBLIC_DOCKER_IMAGE_WILL_GO_HERE'
        Int embeddingCreationCPU = 8
        Int embeddingCreationMemoryGB = 30
        Int embeddingCreationDiskGB = 10
        Int embeddingCreationMaxRetries = 1
        Int embeddingCreationPreemptibleAttempts = 2
        String embeddingCreationGPUType = "nvidia-tesla-t4"
        Int embeddingCreationGPUCount = 1
        String embeddingCreationNvidiaDriverVersion = "470.82.01"
        String embeddingCreationZone = 'us-central1-c'

    }

    String embeddingOutputPathTrimmed = sub(embeddingOutputPath, '/+$', '')

    # Determine which wells should be processed within which shards.
    call determineShards {
        input:
            loadDataWithIllum = loadDataWithIllum,
            modulus = modulus,
            dockerImage = embeddingCreationDockerImage
    }

    # Run embedding creation scattered by shards of multiple wells.
    scatter(shard in determineShards.value) {
        call runEmbeddingCreationScript {
            input:
                shardMetadata = shard,
                loadDataWithIllum = loadDataWithIllum,
                cellCentersPathPrefix = cellCentersPathPrefix,
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
        Array[File] dataWarningsLog = runEmbeddingCreationScript.dataWarningsLog
    }
}

task determineShards {

    input {
        # Specify path to input load_data_with_illum.csv, which contains GCS paths to *.tiff, and illum/*.npy.
        String loadDataWithIllum
        # To determine the shard for a well, perform the modulus operation over the last two digits of the well name
        # or use the numeric value verbatim if the modulus value is zero.
        Int modulus = 24

        # Docker image
        String dockerImage = 'PUBLIC_DOCKER_IMAGE_GOES_HERE'
    }

    String outputFilename = 'shards_metadata.txt'

    command <<<
        # Errors should cause the task to fail, not produce an empty output.
        set -o errexit
        set -o pipefail
        set -o nounset
        # Send a trace of all fully resolved executed commands to stderr.
        set -o xtrace

        python3  /opt/scatter_wells_s3.py \
            --load_data_with_illum_csv_file=~{loadDataWithIllum} \
            --modulus=~{modulus} \
            --output_filename=~{outputFilename}

    >>>

    output {
        Array[String] value = read_lines(outputFilename)
        File outputText = outputFilename
    }

    runtime {
        docker: dockerImage
        maxRetries: 1
        preemptible: 2
    }

}

task runEmbeddingCreationScript {
    input {
        # GCS or S3 path underneath which computed cell centers are stored.
        String shardMetadata
        # Specify path to input load_data_with_illum.csv, which contains GCS paths to *.tiff, and illum/*.npy.
        String loadDataWithIllum
        String cellCentersPathPrefix
        Int cellPatchDim
        Int modelBatchDim
        String tfHubModelPath
        Int tfHubModelInputImageHeight
        Int tfHubModelInputImageWidth
        Int tfHubModelOutputEmbSize

        String dockerImage = 'PUBLIC_DOCKER_IMAGE_GOES_HERE'
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

    command <<<
        # Errors should cause the task to fail, not produce an empty output.
        set -o errexit
        set -o pipefail
        set -o nounset
        # Send a trace of all fully resolved executed commands to stderr.
        set -o xtrace

        mkdir -p ~{workDir}
        cd ~{workDir}

        python3  /opt/embedding_creation.py \
            --shard_metadata='~{shardMetadata}' \
            --cell_center_path_prefix=~{cellCentersPathPrefix} \
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

    >>>

    output {
        File tarOutputs = tarOutputsFile
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
