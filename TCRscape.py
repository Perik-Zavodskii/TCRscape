# https://github.com/Perik-Zavodskii/TCRscape

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.io import mmread
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
import umap.umap_ as umap
import hdbscan

pd.options.mode.chained_assignment = None

def ReadRhapsody(path, sample):
    from scipy.io import mmread
    Barcodes = pd.read_csv(f"{path}/barcodes.tsv.gz", header=None,
                           sep='\t')
    Barcodes = Barcodes.iloc[:, 0]
    Features = pd.read_csv(f"{path}/features.tsv.gz", header=None,
                           sep='\t')
    Features = Features.iloc[:, 0]
    Matrix = mmread(f"{path}/matrix.mtx.gz")
    Matrix = Matrix.todense()
    Rhapsody = pd.DataFrame(Matrix, columns=Barcodes)
    Rhapsody.set_index(Features, inplace=True)
    Rhapsody = Rhapsody.T
    Rhapsody.index.names = ['Cell_Index']
    Rhapsody.reset_index(inplace=True)
    Rhapsody = Rhapsody.assign(Sample=f'{sample}')  # Sample name
    return Rhapsody

def MergeRhapsody(list):
    MergedRhapsody = pd.concat(list)
    MergedRhapsody.set_index(['Cell_Index'], inplace=True)
    MergedRhapsody = MergedRhapsody.fillna(0)
    return MergedRhapsody

def LogNormalize(GEX_list):
    Rhapsody_nosample = GEX_list.drop('Sample', axis=1)
    total_counts = Rhapsody_nosample.sum(axis=1)
    cpm = Rhapsody_nosample.div(total_counts, axis=0) * 1e5
    cpm = cpm.replace({0: 1})
    cpm_log = cpm.map(np.log)
    cpm_log['Sample'] = GEX_list['Sample']
    cpm_log = pd.get_dummies(cpm_log, columns=['Sample'], dtype=int)
    cpm_norm = cpm_log.fillna(0)
    cpm_norm = cpm_norm.astype(float)
    return cpm_norm

def GateCD4(norm_GEX):
    gated = norm_GEX[norm_GEX['CD4'] > 1]
    return gated

def GateCD8(norm_GEX):
    gated = norm_GEX[norm_GEX['CD8A'] > 1]
    return gated

def GateTcells(norm_GEX):
    gated = norm_GEX[(norm_GEX['CD4'] > 1) | (norm_GEX['CD8A'] > 1)]
    return gated

def CountCDR3(gated_GEX, AIRR):
    # Pick GEX-gated cell indices
    gated_labels = gated_GEX.reset_index()
    gated_labels = gated_labels[["Cell_Index"]]
    # Process AIRR matrix
    TCR = AIRR[AIRR["locus"].str.contains("IGH") == False]
    TCR = TCR[TCR["locus"].str.contains("IGK") == False]
    TCR = TCR[TCR["locus"].str.contains("IGL") == False]
    productive = TCR.loc[TCR['productive'] == True]
    # Sort by locus
    productive.sort_values(by='locus', axis=0, ascending=True, inplace=True)
    # Subset Columns
    sort = productive[["cell_id", 'locus', 'cdr3_aa']]
    sort = sort.rename(columns={'cell_id': 'Cell_Index'})
    sort.sort_values(by='Cell_Index', axis=0, ascending=True, inplace=True)
    # Pick individual TCR chains
    sort_A = sort[sort["locus"].str.contains("TRA") == True]
    sort_A['locus_cdr3_aa'] = sort_A[["locus", "cdr3_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    sort_B = sort[sort["locus"].str.contains("TRB") == True]
    sort_B['locus_cdr3_aa'] = sort_B[["locus", "cdr3_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    sort_G = sort[sort["locus"].str.contains("TRG") == True]
    sort_G['locus_cdr3_aa'] = sort_G[["locus", "cdr3_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    sort_D = sort[sort["locus"].str.contains("TRD") == True]
    sort_D['locus_cdr3_aa'] = sort_D[["locus", "cdr3_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    # Alpha-Beta TCR CDR3-typing
    AB = pd.concat([sort_A, sort_B])
    AB.sort_values(by='locus', axis=0, ascending=True, inplace=True)
    AB = AB[["Cell_Index", 'locus_cdr3_aa']]
    AB = AB.groupby("Cell_Index")['locus_cdr3_aa'].apply(lambda x: '___'.join(x.astype(str))).reset_index()
    AB = AB[AB["locus_cdr3_aa"].str.contains("TRA_") == True]
    AB = AB[AB["locus_cdr3_aa"].str.contains("TRB_") == True]
    AB = AB[AB["locus_cdr3_aa"].str.contains("TRA_nan") == False]
    AB = AB[AB["locus_cdr3_aa"].str.contains("TRB_nan") == False]
    CDR3sort_AB = pd.merge(gated_labels, AB)
    CDR3sort_AB.set_index(['Cell_Index'], inplace=True)
    AB_cdr3_counts = CDR3sort_AB.groupby(CDR3sort_AB["locus_cdr3_aa"].tolist(), as_index=False).size()
    AB_cdr3_counts.sort_values(by='size', axis=0, ascending=False, inplace=True)
    AB_cdr3_counts = AB_cdr3_counts.rename(columns={'index': 'CDR3', 'size': 'Number of Cells'})
    # Gamma-Delta TCR CDR3-typing
    GD = pd.concat([sort_G, sort_D])
    GD.sort_values(by='locus', axis=0, ascending=True, inplace=True)
    GD = GD[["Cell_Index", "locus_cdr3_aa"]]
    GD = GD.groupby("Cell_Index")['locus_cdr3_aa'].apply(lambda x: '___'.join(x.astype(str))).reset_index()
    GD = GD[GD["locus_cdr3_aa"].str.contains("TRG_") == True]
    GD = GD[GD["locus_cdr3_aa"].str.contains("TRD_") == True]
    GD = GD[GD["locus_cdr3_aa"].str.contains("TRG_nan") == False]
    GD = GD[GD["locus_cdr3_aa"].str.contains("TRD_nan") == False]
    CDR3sort_GD = pd.merge(gated_labels, GD)
    CDR3sort_GD.set_index(['Cell_Index'], inplace=True)
    GD_cdr3_counts = CDR3sort_GD.groupby(CDR3sort_GD["locus_cdr3_aa"].tolist(), as_index=False).size()
    GD_cdr3_counts.sort_values(by='size', axis=0, ascending=False, inplace=True)
    GD_cdr3_counts = GD_cdr3_counts.rename(columns={'index': 'CDR3', 'size': 'Number of Cells'})
    cdr3_counts = pd.concat([AB_cdr3_counts, GD_cdr3_counts])
    return cdr3_counts

def CDR3pie(gated_GEX, AIRR, min_CDR3, CDR3_top):
    # Pick GEX-gated cell indices
    gated_labels = gated_GEX.reset_index()
    gated_labels = gated_labels[["Cell_Index"]]
    # Process AIRR matrix
    TCR = AIRR[AIRR["locus"].str.contains("IGH") == False]
    TCR = TCR[TCR["locus"].str.contains("IGK") == False]
    TCR = TCR[TCR["locus"].str.contains("IGL") == False]
    productive = TCR.loc[TCR['productive'] == True]
    # Sort by locus
    productive.sort_values(by='locus', axis=0, ascending=True, inplace=True)
    # Subset Columns
    sort = productive[["cell_id", 'locus', 'cdr3_aa']]
    sort = sort.rename(columns={'cell_id': 'Cell_Index'})
    sort.sort_values(by='Cell_Index', axis=0, ascending=True, inplace=True)
    # Build pre-pie chart data
    CDR3 = pd.merge(gated_labels, sort)
    # Count pie chart data
    CDR3_pie = CDR3.groupby(CDR3["cdr3_aa"].tolist(), as_index=False).size()
    CDR3_pie.sort_values(by='size', axis=0, ascending=False, inplace=True)
    CDR3_pie = CDR3_pie.rename(columns={'index': 'CDR3', 'size': 'Percentage of Cells'})
    CDR3_pie = CDR3_pie[CDR3_pie['Percentage of Cells'] >= min_CDR3]
    CDR3_pie[['Percentage of Cells']] = CDR3_pie[['Percentage of Cells']].apply(
        lambda x: x / CDR3_pie[['Percentage of Cells']].sum() * 100, axis=1)
    CDR3_pie.reset_index(drop=True, inplace=True)
    CDR3_pie.loc[CDR3_top:, 'CDR3'] = ''
    plt.pie(CDR3_pie['Percentage of Cells'], labels=CDR3_pie['CDR3'])
    plt.savefig('CDR3-pie chart.png', bbox_inches='tight', dpi=600)
    return CDR3_pie.head(CDR3_top)

def CountClonotypes(gated_GEX, AIRR):
    # Pick GEX-gated cell indices
    gated_labels = gated_GEX.reset_index()
    gated_labels = gated_labels[["Cell_Index"]]
    # Process AIRR matrix
    TCR = AIRR[AIRR["locus"].str.contains("IGH") == False]
    TCR = TCR[TCR["locus"].str.contains("IGK") == False]
    TCR = TCR[TCR["locus"].str.contains("IGL") == False]
    productive = TCR.loc[TCR['productive'] == True]
    # Reconstruct Constant Fragments
    # TRAC
    productive['sequence_aa'] = productive['sequence_aa'].str.replace('IQNPDPAVYQLRDSKSSDKSVCLFTDFD',
                                                                      'IQNPDPAVYQLRDSKSSDKSVCLFTDFDSQTNVSQSKDSDVYITDKTVLDMRSMDFKSNSAVAWSNKSDFACANAFNNSIIPEDTFFPSPESSCDVKLVEKSFETDTNLNFQNLSVIGFRILLLKVAGFNLLMTLRLWSS')
    # TRBC1
    productive['sequence_aa'] = productive['sequence_aa'].str.replace('PEVAVFEPSEA',
                                                                      'PEVAVFEPSEAEISHTQKATLVCLATGFFPDHVELSWWVNGKEVHSGVSTDPQPLKEQPALNDSRYCLSSRLRVSATFWQNPRNHFRCQVQFYGLSENDEWTQDRAKPVTQIVSAEAWGRADCGFTSVSYQQGVLSATILYEILLGKATLYAVLVSALVLMAMVKRKDF')
    productive['sequence_aa'] = productive['sequence_aa'].str.replace('ATILYEILLGKATLYAVLVSALVLMAMVKRKDFEI',
                                                                      'ATILYEILLGKATLYAVLVSALVLMAMVKRKDF')
    productive.sort_values(by='locus', axis=0, ascending=True, inplace=True)
    # Subset Columns
    sort = productive[["cell_id", 'locus', 'sequence_aa', 'cdr3_aa']]
    sort = sort.rename(columns={'cell_id': 'Cell_Index'})
    sort.sort_values(by='Cell_Index', axis=0, ascending=True, inplace=True)
    sort['sequence_aa'] = sort['sequence_aa'].apply(lambda x: x.rsplit('*', maxsplit=1)[-1])
    sort['sequence_aa'] = sort['sequence_aa'].str.replace('^.*?[M]', 'M', regex=True)
    # Pick individual TCR chains
    sort_A = sort[sort["locus"].str.contains("TRA") == True]
    sort_A['locus_sequence_aa'] = sort_A[["locus", "sequence_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    sort_B = sort[sort["locus"].str.contains("TRB") == True]
    sort_B['locus_sequence_aa'] = sort_B[["locus", "sequence_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    sort_G = sort[sort["locus"].str.contains("TRG") == True]
    sort_G['locus_sequence_aa'] = sort_G[["locus", "sequence_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    sort_D = sort[sort["locus"].str.contains("TRD") == True]
    sort_D['locus_sequence_aa'] = sort_D[["locus", "sequence_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    # Alpha-Beta TCR clonotyping
    AB = pd.concat([sort_A, sort_B])
    AB.sort_values(by='locus', axis=0, ascending=True, inplace=True)
    AB = AB[["Cell_Index", 'locus_sequence_aa']]
    AB = AB.groupby("Cell_Index")['locus_sequence_aa'].apply(lambda x: '___'.join(x.astype(str))).reset_index()
    AB = AB[AB["locus_sequence_aa"].str.contains("TRA_") == True]
    AB = AB[AB["locus_sequence_aa"].str.contains("TRB_") == True]
    AB = AB[AB["locus_sequence_aa"].str.contains("TRA_nan") == False]
    AB = AB[AB["locus_sequence_aa"].str.contains("TRB_nan") == False]
    clonosort_AB = pd.merge(gated_labels, AB)
    clonosort_AB.set_index(['Cell_Index'], inplace=True)
    AB_counts = clonosort_AB.groupby(clonosort_AB["locus_sequence_aa"].tolist(), as_index=False).size()
    AB_counts.sort_values(by='size', axis=0, ascending=False, inplace=True)
    AB_counts = AB_counts.rename(columns={'index': 'TCR', 'size': 'Number of Cells'})
    # Gamma-Delta TCR clonotyping
    GD = pd.concat([sort_G, sort_D])
    GD.sort_values(by='locus', axis=0, ascending=True, inplace=True)
    GD = GD[["Cell_Index", "locus_sequence_aa"]]
    GD = GD.groupby("Cell_Index")['locus_sequence_aa'].apply(lambda x: '___'.join(x.astype(str))).reset_index()
    GD = GD[GD["locus_sequence_aa"].str.contains("TRG_") == True]
    GD = GD[GD["locus_sequence_aa"].str.contains("TRD_") == True]
    GD = GD[GD["locus_sequence_aa"].str.contains("TRG_nan") == False]
    GD = GD[GD["locus_sequence_aa"].str.contains("TRD_nan") == False]
    clonosort_GD = pd.merge(gated_labels, GD)
    clonosort_GD.set_index(['Cell_Index'], inplace=True)
    GD_counts = clonosort_GD.groupby(clonosort_GD["locus_sequence_aa"].tolist(), as_index=False).size()
    GD_counts.sort_values(by='size', axis=0, ascending=False, inplace=True)
    GD_counts = GD_counts.rename(columns={'index': 'TCR', 'size': 'Number of Cells'})
    counts = pd.concat([AB_counts, GD_counts])
    return counts

def TCRscape(norm_GEX, AIRR, features_to_cluster, min_clones):
    TCRscape = norm_GEX[features_to_cluster]
    TCRscape = TCRscape.astype(float)
    TCRscape.reset_index(drop=False, inplace=True)
    # Pick GEX-gated cell indices
    gated_labels = TCRscape.reset_index()
    gated_labels = gated_labels[["Cell_Index"]]
    # Process AIRR matrix
    TCR = AIRR[AIRR["locus"].str.contains("IGH") == False]
    TCR = TCR[TCR["locus"].str.contains("IGK") == False]
    TCR = TCR[TCR["locus"].str.contains("IGL") == False]
    productive = TCR.loc[TCR['productive'] == True]
    # Reconstruct Constant Fragments
    productive['sequence_aa'] = productive.loc[:, 'sequence_aa']
    # TRAC
    productive['sequence_aa'] = productive['sequence_aa'].str.replace('IQNPDPAVYQLRDSKSSDKSVCLFTDFD',
                                                                      'IQNPDPAVYQLRDSKSSDKSVCLFTDFDSQTNVSQSKDSDVYITDKTVLDMRSMDFKSNSAVAWSNKSDFACANAFNNSIIPEDTFFPSPESSCDVKLVEKSFETDTNLNFQNLSVIGFRILLLKVAGFNLLMTLRLWSS')
    # TRBC1
    productive['sequence_aa'] = productive['sequence_aa'].str.replace('PEVAVFEPSEA',
                                                                      'PEVAVFEPSEAEISHTQKATLVCLATGFFPDHVELSWWVNGKEVHSGVSTDPQPLKEQPALNDSRYCLSSRLRVSATFWQNPRNHFRCQVQFYGLSENDEWTQDRAKPVTQIVSAEAWGRADCGFTSVSYQQGVLSATILYEILLGKATLYAVLVSALVLMAMVKRKDF')
    productive['sequence_aa'] = productive['sequence_aa'].str.replace('ATILYEILLGKATLYAVLVSALVLMAMVKRKDFEI',
                                                                      'ATILYEILLGKATLYAVLVSALVLMAMVKRKDF')
    # Sort by locus
    productive.sort_values(by='locus', axis=0, ascending=True, inplace=True)
    # Subset Columns
    sort = productive[["cell_id", 'locus', 'sequence_aa', 'cdr3_aa']]
    sort = sort.rename(columns={'cell_id': 'Cell_Index'})
    sort.sort_values(by='Cell_Index', axis=0, ascending=True, inplace=True)
    sort['sequence_aa'] = sort['sequence_aa'].apply(lambda x: x.rsplit('*', maxsplit=1)[-1])
    sort['sequence_aa'] = sort['sequence_aa'].str.replace('^.*?[M]', 'M', regex=True)
    # Pick individual TCR chains
    sort_A = sort[sort["locus"].str.contains("TRA") == True]
    sort_A['locus_sequence_aa'] = sort_A[["locus", "sequence_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    sort_B = sort[sort["locus"].str.contains("TRB") == True]
    sort_B['locus_sequence_aa'] = sort_B[["locus", "sequence_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    sort_G = sort[sort["locus"].str.contains("TRG") == True]
    sort_G['locus_sequence_aa'] = sort_G[["locus", "sequence_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    sort_D = sort[sort["locus"].str.contains("TRD") == True]
    sort_D['locus_sequence_aa'] = sort_D[["locus", "sequence_aa"]].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1)
    # Alpha-Beta TCR clonotyping
    AB = pd.concat([sort_A, sort_B])
    AB.sort_values(by='locus', axis=0, ascending=True, inplace=True)
    AB = AB[["Cell_Index", 'locus_sequence_aa']]
    AB = AB.groupby("Cell_Index")['locus_sequence_aa'].apply(lambda x: '___'.join(x.astype(str))).reset_index()
    AB = AB[AB["locus_sequence_aa"].str.contains("TRA_") == True]
    AB = AB[AB["locus_sequence_aa"].str.contains("TRB_") == True]
    AB = AB[AB["locus_sequence_aa"].str.contains("TRA_nan") == False]
    AB = AB[AB["locus_sequence_aa"].str.contains("TRB_nan") == False]
    clonosort_AB = pd.merge(gated_labels, AB)
    clonosort_AB.set_index(['Cell_Index'], inplace=True)
    AB_counts = clonosort_AB.groupby(clonosort_AB["locus_sequence_aa"].tolist(), as_index=False).size()
    AB_counts.sort_values(by='size', axis=0, ascending=False, inplace=True)
    AB_counts = AB_counts.rename(columns={'index': 'AB TCR', 'size': 'Number of Cells'})
    # Gamma-Delta TCR clonotyping
    GD = pd.concat([sort_G, sort_D])
    GD.sort_values(by='locus', axis=0, ascending=True, inplace=True)
    GD = GD[["Cell_Index", "locus_sequence_aa"]]
    GD = GD.groupby("Cell_Index")['locus_sequence_aa'].apply(lambda x: '___'.join(x.astype(str))).reset_index()
    GD = GD[GD["locus_sequence_aa"].str.contains("TRG_") == True]
    GD = GD[GD["locus_sequence_aa"].str.contains("TRD_") == True]
    GD = GD[GD["locus_sequence_aa"].str.contains("TRG_nan") == False]
    GD = GD[GD["locus_sequence_aa"].str.contains("TRD_nan") == False]
    clonosort_GD = pd.merge(gated_labels, GD)
    clonosort_GD.set_index(['Cell_Index'], inplace=True)
    GD_counts = clonosort_GD.groupby(clonosort_GD["locus_sequence_aa"].tolist(), as_index=False).size()
    GD_counts.sort_values(by='size', axis=0, ascending=False, inplace=True)
    GD_counts = GD_counts.rename(columns={'index': 'GD TCR', 'size': 'Number of Cells'})
    AB_counts = AB_counts.rename(columns={'AB TCR': 'locus_sequence_aa'})
    GD_counts = GD_counts.rename(columns={'GD TCR': 'locus_sequence_aa'})
    # Merge clonotype counts
    counts = pd.concat([AB_counts, GD_counts])
    counts = counts.reset_index()
    counts.drop('index', axis=1, inplace=True)
    counts.set_index(['locus_sequence_aa'], inplace=True)
    ab_index = AB.reset_index()
    gd_index = GD.reset_index()
    index = pd.concat([ab_index, gd_index])
    index.drop('index', axis=1, inplace=True)
    index.set_index(['locus_sequence_aa'], inplace=True)
    clonosort = pd.merge(index, counts, left_index=True, right_index=True)
    clonosort = clonosort.reset_index()
    clonosort = clonosort[clonosort['Number of Cells'] >= min_clones]
    TCRscape = pd.merge(TCRscape, clonosort)
    TCRscape = TCRscape.rename(columns={'locus_sequence_aa': 'Clonotype'})
    TCRscape['TCR_Type'] = TCRscape['Clonotype'].str.replace('TRA.*', '1', regex=True)
    TCRscape['TCR_Type'] = TCRscape['TCR_Type'].str.replace('TRD.*', '0', regex=True)
    TCRscape["TCR_Type"] = TCRscape["TCR_Type"].astype(int)
    TCRscape = pd.get_dummies(TCRscape, columns=['Clonotype'], dtype=int)
    TCRscape.set_index(['Cell_Index'], inplace=True)
    # Filter productive chains from AIRR matrix
    alpha = AIRR[(AIRR['locus'] == 'TRA') & (AIRR['productive'] == True)]
    beta  = AIRR[(AIRR['locus'] == 'TRB') & (AIRR['productive'] == True)]
    gamma = AIRR[(AIRR['locus'] == 'TRG') & (AIRR['productive'] == True)]
    delta = AIRR[(AIRR['locus'] == 'TRD') & (AIRR['productive'] == True)]
    # Keep only top UMI chain per cell
    alpha = alpha.sort_values(['cell_id', 'umi_count'], ascending=False).drop_duplicates('cell_id')
    beta  = beta.sort_values(['cell_id', 'umi_count'], ascending=False).drop_duplicates('cell_id')
    gamma = gamma.sort_values(['cell_id', 'umi_count'], ascending=False).drop_duplicates('cell_id')
    delta = delta.sort_values(['cell_id', 'umi_count'], ascending=False).drop_duplicates('cell_id')
    # Rename for clarity and merge αβ and γδ chains separately
    ab = pd.merge(
        alpha.rename(columns={'v_call': 'TRAV', 'j_call': 'TRAJ'}),
        beta.rename(columns={'v_call': 'TRBV', 'd_call': 'TRBD', 'j_call': 'TRBJ'}),
        on='cell_id', how='inner'
    )
    gd = pd.merge(
        gamma.rename(columns={'v_call': 'TRGV', 'j_call': 'TRGJ'}),
        delta.rename(columns={'v_call': 'TRDV', 'd_call': 'TRDD', 'j_call': 'TRDJ'}),
     on='cell_id', how='inner'
    )
    # Add binary TCR type label
    ab['TCR_type'] = 1  # alpha-beta
    gd['TCR_type'] = 0  # gamma-delta
    # Combine the two TCR types
    tcr_df = pd.concat([ab, gd], ignore_index=True)
    # One-hot encode V, D, J genes only relevant to TCR type
    ab_vdj = pd.get_dummies(
      ab[['TRAV', 'TRAJ', 'TRBV', 'TRBD', 'TRBJ']],
      prefix=['TRAV', 'TRAJ', 'TRBV', 'TRBD', 'TRBJ']
    )
    gd_vdj = pd.get_dummies(
        gd[['TRGV', 'TRGJ', 'TRDV', 'TRDD', 'TRDJ']],
       prefix=['TRGV', 'TRGJ', 'TRDV', 'TRDD', 'TRDJ']
    )
    # Add TCR_type to each encoded set
    ab_vdj['TCR_type'] = 1
    gd_vdj['TCR_type'] = 0
    # Index on cell_id
    ab_vdj.index = ab['cell_id']
    gd_vdj.index = gd['cell_id']
    # Final VDJ one-hot matrix
    vdj_onehot = pd.concat([ab_vdj, gd_vdj])
    vdj_onehot = vdj_onehot.fillna(0)
    vdj_onehot.index.name = 'Cell_Index'
    vdj_onehot = vdj_onehot.map(lambda x: 1 if x else 0)
    TCRscape = TCRscape.join(vdj_onehot, how='inner')
    return TCRscape
