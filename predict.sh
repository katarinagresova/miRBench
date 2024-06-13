#### START OF USER INPUT ####

# paths to downloaded datasets - number for ratio and extension is added automatically 
declare -a INPUT_DATA=(
    "$HOME/miRBench/data/Helwak_2013/miRNA_test_set_"
    "$HOME/miRBench/data/Klimentova_2022/miRNA_test_set_"
    "$HOME/miRBench/data/Hejret_2023/miRNA_test_set_"
)

# datasets are available in three ratios - 1:1, 1:10 and 1:100. 
declare -a RATIOS=(
    "1"
    "10"
    "100"
)

declare -a TOOLS=(
    "seed"
    "TargetScanCnn"
    "miRBind"
    "HejretMirnaCnn"
    "cofold"
    "YangAttention"
    "cnnMirTarget"
    "TargetNet"
)

#### END OF USER INPUT ####

EXTENSION=".tsv"
miRNA_COLUMN='noncodingRNA'
gene_COLUMN='gene'

# iterate over the array of input data
for i in "${!INPUT_DATA[@]}"; do

    DATASETS=""

    for ratio in "${!RATIOS[@]}"; do

        INPUT_FILE=${INPUT_DATA[$i]}${RATIOS[$ratio]}${EXTENSIONS[$i]}
        echo "Executing for ${INPUT_FILE}"

        DATASETS=${DATASETS}" "${INPUT_FILE}

        SUFFIXES=""

        # iterate over the array of TOOLS
        for j in "${!TOOLS[@]}"; do
            echo "Predicting using ${TOOLS[$j]}"
            python $HOME/miRBench/src/miRBench/tools/${TOOLS[$j]}.py \
                --input ${INPUT_DATA[$i]}${RATIOS[$ratio]}${SUFFIXES}${EXTENSION} \
                --miRNA_column ${miRNA_COLUMN} \
                --gene_column ${gene_COLUMN} \
                --output ${INPUT_DATA[$i]}${RATIOS[$ratio]}${SUFFIXES}"_"${TOOLS[$j]}${EXTENSION}

            # delete previous output file
            if [ $j -gt 0 ]; then
                rm ${INPUT_DATA[$i]}${RATIOS[$ratio]}${SUFFIXES}${EXTENSION}
            fi

            # accumulate the suffixes of the algorithms
            SUFFIXES=${SUFFIXES}"_"${TOOLS[$j]}
        done

        # rename the last output file to have the same name as the input file + '_predictions'
        mv ${INPUT_DATA[$i]}${RATIOS[$ratio]}${SUFFIXES}${EXTENSION} ${INPUT_DATA[$i]}${RATIOS[$ratio]}"_predictions"${EXTENSION}

    done
done