Each directory is a protein analyzed in the paper:
Adam J. Riesselman, John B. Ingraham, and Debora S. Marks. Deep generative models of genetic variation capture the effects of mutations. Nature Methods, 15(10):816–822, 10 2018. ISSN 15487105. doi: 10.1038/s41592-018-0138-4.

###Metadata:
41 separate folders with a unique DMS.
15 with Protein Data Bank structure
27 with AlphaFold structure
21 with generated ColabFold structure: [AlphaFold2_advanced](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/AlphaFold2_advanced.ipynb#scrollTo=rowN0bVYLe9n)

###This directory /data contains 3 types of files:
1. this file
2. protein directory
3. jupyter notebook file that prints amino acid sequences from deep mutational scan csv file

###Each protein directory contains at least 3 files:
1. Deep mutational scan: .csv file starting with output, collected from Riesselman supplementary table 2.
2. Alignment: .a2m files, collected from Reisselman attached [github](https://github.com/debbiemarkslab/EVcouplings)
3. PDB ids: pdb.txt which contains likely pdb ids, including the associated name and notes on how the
	modeled protein sequence compares to the provided deep mutational scan and alignment sequences. The notes include
	information on comparison of sequence length, amino acids, gaps.

###The format of pdb.txt is:
####First 13 Lines:
	Important Note: [NOTE]
	
	[CITATION]
	[NOTES_ON_PAPER]
	
	DMS Seq:
	[DMS_SEQUENCE]
	Alignment Seq:
	[ALIGNMENT_SEQUENCE]

	DMS Seq Range: [DEEP_MUTATIONAL_SCAN_SEQUENCE_RANGE], ALIGNMENT Seq Range: [FIRST_ALIGNMENT_SEQUENCE_RANGE]
	[NOTES_ON_DIFFERENCES]
	[ALIGNMENT_DATA_ON_DMS_AND_.A2M] (data from https://web.expasy.org/sim/, with parameters: Gap open penalty: 12, Gap extension penalty 4, Comparison matrix BLOSUM62)
####For each possible PDB id:
	-[PDB_ID]: [NAME]
	Method: [MOLECULAR_STRUCTURE_CHARACTERIZATION_METHOD]
	Seq Length: [LENGTH]
	Alignment [SEQUENCE_MATCH_INFO] (compared against first alignment sequence)
	DMS [SEQUENCE_MATCH_INFO] (compared against DMS sequence)
	[NOTES_ON_DIFFERENCES]
####For the EBI Protein Similarity Search (https://www.ebi.ac.uk/Tools/sss/fasta/):
	+DMS Seq EBI Query:
	[PROTEIN_NAME]
	Swiss Prot ID: [ID]
	Length: [LENGTH]
	Identities: [PERCENTAGE]
	Positives: [PERCENTAGE]

	+Alignment Seq EBI Query:
	[PROTEIN_NAME]
	Swiss Prot ID: [ID]
	Length: [LENGTH]
	Identities: [PERCENTAGE]
	Positives: [PERCENTAGE]

###Directories may contain these additional files:
4. PDB ids: .csv file starting with pdb, collected from Reisselman supplementary table 8,
5. PDB File: [4_CHAR_CODE].pdb from protein data bank, If I have high confidence that the .pdb sequence is correct
	(because the sequence matches the deep mutational scan and alignment sequence, including length, amino acids, gaps).



###This data is also used in:
Language models enable zero-shot prediction of the effects of mutations on protein function
Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, Alexander Rives
bioRxiv 2021.07.09.450648; doi: https://doi.org/10.1101/2021.07.09.450648

For the purpose (as noted in C.4 Validation and test set):
The single-mutation validation set consists of the following deep mutational scans: AMIE_PSEAE_Whitehead, BG505_env_Bloom2018, BLAT_ECOLX_Ranganathan2015, BRCA1_HUMAN_RING, DLG4_RAT_Ranganathan2012, GAL4_YEAST_Shendure2015, POLG_HCVJF_Sun2014, SUMO1_HUMAN_Roth2017, TIM_SULSO_b0, UBC9_HUMAN_Roth2017, KKA2_KLEPN_Mikkelsen2014.

For ablations studies with multiple mutations the following dataset is used: PABP_YEAST_Fields2013-doubles

The test set consists of the following deep mutational scans: B3VI55_LIPSTSTABLE, B3VI55_LIPST_Whitehead2015, BF520_env_Bloom2018, BG_STRSQ_hmmerbit, BLAT_ECOLX_Ostermeier2014, BLAT_ECOLX_Palzkill2012, BLAT_ECOLX_Tenaillon2013, BRCA1_HUMAN_BRCT, CALM1_HUMAN_Roth2017, HG_FLU_Bloom2016, HIS7_YEAST_Kondrashov2017, HSP82_YEAST_Bolon2016, IF1_ECOLI_Kishony, MK01_HUMAN_Johannessen, MTH3_HAEAESTABILIZED_Tawfik2015, P84126_THETH_b0, PABP_YEAST_Fields2013-singles, PA_FLU_Sun2015, POL_HV1N5-CA_Ndungu2014, PTEN_HUMAN_Fowler2018, RASH_HUMAN_Kuriyan, RL401_YEAST_Bolon2013, RL401_YEAST_Bolon2014, RL401_YEAST_Fraser2016, TIM_THEMA_b0, TPK1_HUMAN_Roth2017, TPMT_HUMAN_Fowler2018, UBE4B_MOUSE_Klevit2013-singles, YAP1_HUMAN_Fields2012-singles.