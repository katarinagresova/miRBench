import argparse
import pandas as pd
import RNA

def predict(data, miRNA_column, gene_column):
    return -1 * RNA.cofold(data[miRNA_column] + "&" + data[gene_column])[1]
    

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RNAcofold prediction.')
    # Path to the input file 
    parser.add_argument('--input', type=str, help='Path to the input file - miRNA and a gene sequence in a tab-separated format.')
    # Name of column containing miRNA sequences
    parser.add_argument('--miRNA_column', type=str, help='Name of the column containing miRNA sequences')
    # Name of column containing gene sequences
    parser.add_argument('--gene_column', type=str, help='Name of the column containing gene sequences')
    # Path to the output file
    parser.add_argument('--output', type=str, help='Path to the output file')
    # Parse the arguments
    args = parser.parse_args()

    # Read the input file
    data = pd.read_csv(args.input, sep='\t')

    data['cofold'] = data.apply(lambda row: predict(row, args.miRNA_column, args.gene_column), axis=1)
    data.to_csv(args.output, sep='\t', index=False)