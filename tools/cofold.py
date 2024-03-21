import pandas as pd
import RNA

from utils import parse_args

def predict(data, miRNA_column, gene_column):
    return -1 * RNA.cofold(data[miRNA_column] + "&" + data[gene_column])[1]
    

if __name__ == '__main__':
    args = parse_args('RNAcofold')

    data = pd.read_csv(args.input, sep='\t')

    data['cofold'] = data.apply(lambda row: predict(row, args.miRNA_column, args.gene_column), axis=1)
    data.to_csv(args.output, sep='\t', index=False)