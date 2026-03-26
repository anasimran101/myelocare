#!/bin/bash

# this script is for merging orginal dataset with the syntheic datasets to generate a new combined dataset for training the model

# Def Paths

ORIGINAL_DATASET_PATH="datasets/MMDB/data/detection/train"
SYNTHETIC_DATASET_PATH="datasets/synthetic_datasets/synthetic_v1"
MERGED_DATASET_PATH="datasets/merged_data/"




# load the PATH passed as an argument to the script, if not use the default paths defined above, args are defined as arg=value
# example: ./merger_datasets.sh ORIGINAL_DATASET_PATH=datasets/MMDB/data/detection/train SYNTHETIC_DATASET_PATH=datasets/synthetic_data/synthetic_v1 MERGED_DATASET_PATH=datasets/merged_data/

for ARG in "$@"; do
    case $ARG in
        ORIGINAL=*)
            ORIGINAL_DATASET_PATH="${ARG#*=}"
            shift
            ;;
        SYNTHETIC=*)
            SYNTHETIC_DATASET_PATH="${ARG#*=}"
            shift
            ;;
        MERGED=*)
            MERGED_DATASET_PATH="${ARG#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [ORIGINAL=path_to_original_dataset] [SYNTHETIC=path_to_synthetic_dataset] [MERGED=path_to_merged_dataset]"
            exit 0
            ;;
        *)
            echo "Invalid argument: $ARG"
            exit 1
            ;;
    esac
done


# Check if the script is being run from the correct directory
if [[ "$(basename "$(pwd)")" != "myelocare" ]]; then
  echo "Please run this script from the myelocare directory"
  exit 1
fi



#check if original dataset path exists
if [ ! -d "$ORIGINAL_DATASET_PATH" ]; then
  echo "Original dataset path does not exist: $ORIGINAL_DATASET_PATH"
  exit 1
fi  
#check if synthetic dataset path exists
if [ ! -d "$SYNTHETIC_DATASET_PATH" ]; then
  echo "Synthetic dataset path does not exist: $SYNTHETIC_DATASET_PATH"
  exit 1
fi

#check if merged dataset path exists, if yes delete it and create a new one
if [ -d "$MERGED_DATASET_PATH" ]; then
    rmdir -rf "$MERGED_DATASET_PATH"
fi
mkdir -p "$MERGED_DATASET_PATH"

#check if images and labels directories exist in the merged dataset path, if not create them
if [ ! -d "$MERGED_DATASET_PATH/images" ]; then
  mkdir -p "$MERGED_DATASET_PATH/images"
fi  
if [ ! -d "$MERGED_DATASET_PATH/labels" ]; then
  mkdir -p "$MERGED_DATASET_PATH/labels"
fi 

cp -r "$ORIGINAL_DATASET_PATH"/images "$MERGED_DATASET_PATH/images"
cp -r "$ORIGINAL_DATASET_PATH"/labels "$MERGED_DATASET_PATH/labels"

cp -r "$SYNTHETIC_DATASET_PATH"/images/* "$MERGED_DATASET_PATH"/images/
cp -r "$SYNTHETIC_DATASET_PATH"/labels/* "$MERGED_DATASET_PATH"/labels/


# make a readme with the rich details of the merged dataset

echo "Merged Dataset Details" > "$MERGED_DATASET_PATH/README.txt"
echo "Original Dataset: $ORIGINAL_DATASET_PATH" >> "$MERGED_DATASET_PATH/README.txt"
echo "Synthetic Dataset: $SYNTHETIC_DATASET_PATH" >> "$MERGED_DATASET_PATH/README.txt"
echo "Number of images in original dataset: $(ls "$ORIGINAL_DATASET_PATH/images" | wc -l)" >> "$MERGED_DATASET_PATH/README.txt"
echo "Number of images in synthetic dataset: $(ls "$SYNTHETIC_DATASET_PATH/images" | wc -l)" >> "$MERGED_DATASET_PATH/README.txt"
echo "Total number of images in merged dataset: $(ls "$MERGED_DATASET_PATH/images" | wc -l)" >> "$MERGED_DATASET_PATH/README.txt"



echo "Datasets merged successfully! Merged dataset path: $MERGED_DATASET_PATH"