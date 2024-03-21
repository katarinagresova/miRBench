import pandas as pd

from utils import parse_args

def rev_compl(st):
    nn = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join(nn[n] for n in reversed(st))

def seeds_8mer(miRNA):
    return [
        'A' + rev_compl(miRNA[1:8]) # 8mer - full complementarity on positions 2-8 and A on the position 1
    ]

def seeds_7mer(miRNA):
    return [
        rev_compl(miRNA[1:8]), # 7mer-m8 - full complementarity on positions 2-8
        'A' + rev_compl(miRNA[1:7]) # 7mer-A1 - full complementarity on positions 2-7 and A on the position 1
    ]

def seeds_6mer(miRNA):
    return [
        rev_compl(miRNA[1:7]), # 6mer - full complementarity on positions 2-7
        rev_compl(miRNA[2:8]), # 6mer-m8 - full complementarity on positions 3-8
        'A' + rev_compl(miRNA[1:6]) # 6mer-A1 - full complementarity on positions 2-6 and A on the position 1
    ]

def seeds_6mer_bulge_or_mismatch(miRNA):
    mers = []
    mers.append(rev_compl(miRNA[1:7]))
    for pos in range(1, 7):
        for nt in ['A', 'C', 'G', 'T']:
            # bulges
            mers.append(
                rev_compl(miRNA[1:7])[:pos] + nt + rev_compl(miRNA[1:7])[pos:]
            )
            # mismatches
            mers.append(
                rev_compl(miRNA[1:7])[:pos] + nt + rev_compl(miRNA[1:7])[pos+1:]
            )
    mers.append(rev_compl(miRNA[2:8]))
    for pos in range(2, 8):
        for nt in ['A', 'C', 'G', 'T']:
            mers.append(
                rev_compl(miRNA[2:8])[:pos] + nt + rev_compl(miRNA[2:8])[pos:]
            )
            mers.append(
                rev_compl(miRNA[2:8])[:pos] + nt + rev_compl(miRNA[2:8])[pos+1:]
            )
    mers.append('A' + rev_compl(miRNA[1:6]))
    for pos in range(1, 6):
        for nt in ['A', 'C', 'G', 'T']:
            mers.append(
                'A' + rev_compl(miRNA[1:6])[:pos] + nt + rev_compl(miRNA[1:6])[pos:]
            )
            mers.append(
                'A' + rev_compl(miRNA[1:6])[:pos] + nt + rev_compl(miRNA[1:6])[pos+1:]
            )

    return list(set(mers))

def classify_site(gene, miRNA, get_seeds):

    seeds = get_seeds(miRNA)

    for seq in seeds:
        if seq in gene:
            return 1
        
    return 0

if __name__ == '__main__':
    args = parse_args('Seed')

    # Read the input file
    data = pd.read_csv(args.input, sep='\t')

    data['kmer8'] = data.apply(lambda x: classify_site(x[args.gene_column], x[args.miRNA_column], seeds_8mer), axis=1)
    data['kmer7'] = data.apply(lambda x: classify_site(x[args.gene_column], x[args.miRNA_column], seeds_7mer), axis=1)
    data['kmer6'] = data.apply(lambda x: classify_site(x[args.gene_column], x[args.miRNA_column], seeds_6mer), axis=1)
    data['kmer6_bulge_or_mismatch'] = data.apply(lambda x: classify_site(x[args.gene_column], x[args.miRNA_column], seeds_6mer_bulge_or_mismatch), axis=1)

    data.to_csv(args.output, sep='\t', index=False)

