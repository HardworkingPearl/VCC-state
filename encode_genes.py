#!/usr/bin/env python3
import os
import requests
import time
import torch
import logging
from pathlib import Path
from Bio import SeqIO
import pickle
import esm  # pip install fair-esm
# For Evo2: you may need to pip install evo2 or use transformers depending on version
# pip install evo2  (or from GitHub)  :contentReference[oaicite:3]{index=3}
from tqdm import tqdm 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# pid=os.getpid()
# import signal;os.kill(pid,signal.SIGKILL)
# -------- CONFIGURATION --------
with open("/home/absking/scratch/vcc/pert_genes.pkl", "rb") as f:
    loaded_set = pickle.load(f)

# Convert to list:
GENES = list(loaded_set)
# GENES = [
#   'PNMA8A', 'FST', 'RSU1', 'C2orf16', 'SQSTM1', 'SUV39H2', 'UBE4A', 'ZNF705D', 'ZNF706', 'TIAM1', 'MOAP1', 'RXFP3', 'ADAMTS7', 'DCBLD1', 'TCHH', 'PGF', 'CEP95', 'ZNF106', 
# ]

OUTPUT_FASTA_PROTEIN = "genes_proteins.fasta"
OUTPUT_FASTA_NUC = "genes_nucleotides.fasta"

EMBED_OUT_PROTEIN = "genes_protein_embeddings.pt"
EMBED_OUT_NUC = "genes_nuc_embeddings.pt"

# UniProt REST API (protein sequences)
UNIPROT_FASTA_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?"
    "format=fasta&query=gene:{gene}%20AND%20organism_id:9606%20AND%20(reviewed:true)"
)

# Ensembl or other source for nucleotide sequences – placeholder URL
# You’ll need to implement a fetch function for nucleotide FASTA.
ENSEMBL_CDS_URL = "https://rest.ensembl.org/sequence/id/{transcript_id}?type=cds;format=fasta"

# -------- STEP 1A: Download protein FASTA --------
def fetch_uniprot_fasta(gene_symbol):
    url = UNIPROT_FASTA_URL.format(gene=gene_symbol)
    logger.info(f"Fetching protein FASTA for gene {gene_symbol}")
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Error fetching {gene_symbol} protein: HTTP {r.status_code}")
    return r.text

def build_protein_fasta(gene_list, output_path):
    with open(output_path, "w") as fw:
        for gene in gene_list:
            try:
                fasta = fetch_uniprot_fasta(gene)
                fw.write(fasta)
                time.sleep(1.0)  # rate limiting
            except Exception as e:
                logger.warning(f"Skipping gene {gene} (protein): {e}")
    logger.info(f"Protein FASTA written to {output_path}")

# -------- STEP 1B: Download nucleotide FASTA --------
ENSEMBL_REST_BASE = "https://rest.ensembl.org"

def fetch_nucleotide_fasta(gene_symbol,
                          species="homo_sapiens",
                          seq_type="cds",
                          fallback_types=None):
    """
    Fetch a nucleotide FASTA sequence for a given gene symbol, resolving aliases via Ensembl,
    and stripping transcript version suffix if present (e.g., ENST00000335288.5 → ENST00000335288).
    Parameters:
      gene_symbol : str — Input symbol (could be alias).
      species     : str — Ensembl species name, default "homo_sapiens".
      seq_type    : str — One of "cds", "cdna", "genomic". Default "cds".
      fallback_types : list of str — additional types to try if first fails.
    Returns:
      fasta_str   : str — FASTA formatted nucleotide sequence.
    Raises:
      RuntimeError on lookup or sequence fetch failure.
    """
    if fallback_types is None:
        fallback_types = ["cdna", "genomic"]

    # STEP 0: Resolve aliases → gene/transcript ID
    alias_url = f"{ENSEMBL_REST_BASE}/xrefs/symbol/{species}/{gene_symbol}?content-type=application/json"
    logger.info(f"Resolving symbol / alias for {gene_symbol}")
    r_alias = requests.get(alias_url, headers={"Content-Type":"application/json"}, timeout=30)
    if not r_alias.ok:
        raise RuntimeError(f"Alias lookup failed for {gene_symbol}: HTTP {r_alias.status_code}")
    xrefs = r_alias.json()
    if not xrefs:
        logger.warning(f"No xrefs found for symbol {gene_symbol}. Proceeding with original symbol.")
        resolved_symbol = gene_symbol
        ens_gene_id = None
    else:
        gene_hits = [x for x in xrefs if x.get("type") == "gene"]
        if gene_hits:
            chosen = gene_hits[0]
        else:
            chosen = xrefs[0]
        resolved_symbol = chosen.get("display_id") or gene_symbol
        ens_gene_id = chosen.get("id")
        logger.info(f"Resolved {gene_symbol} → symbol {resolved_symbol}, Ensembl ID {ens_gene_id}")
    # time.sleep(0.2)

    # STEP 1: Get transcript ID
    if ens_gene_id is None:
        lookup_url = f"{ENSEMBL_REST_BASE}/lookup/symbol/{species}/{resolved_symbol}?content-type=application/json"
        logger.info(f"Lookup gene/transcript for symbol {resolved_symbol}")
        r_lookup = requests.get(lookup_url, headers={"Content-Type":"application/json"}, timeout=30)
        if not r_lookup.ok:
            raise RuntimeError(f"Lookup failed for {resolved_symbol}: HTTP {r_lookup.status_code}")
        info = r_lookup.json()
        transcript_id = info.get("canonical_transcript") or (info.get("transcript") or [None])[0]
        if not transcript_id:
            raise RuntimeError(f"No transcript found for gene symbol {resolved_symbol}")
    else:
        lookup_url = f"{ENSEMBL_REST_BASE}/lookup/id/{ens_gene_id}?content-type=application/json&expand=1"
        logger.info(f"Lookup info for Ensembl gene ID {ens_gene_id}")
        r_lookup = requests.get(lookup_url, headers={"Content-Type":"application/json"}, timeout=30)
        if not r_lookup.ok:
            raise RuntimeError(f"Lookup failed for ID {ens_gene_id}: HTTP {r_lookup.status_code}")
        info = r_lookup.json()
        transcript_id = info.get("canonical_transcript")
        if not transcript_id:
            transcripts = info.get("Transcript") or []
            if transcripts:
                transcript_id = transcripts[0].get("id")
        if not transcript_id:
            raise RuntimeError(f"No transcript found for Ensembl gene ID {ens_gene_id}")

    logger.info(f"Selected transcript ID: {transcript_id}")

    # Strip version suffix if present (e.g., “.5”)
    if "." in transcript_id:
        base_id = transcript_id.split(".")[0]
        logger.info(f"Stripping version suffix: {transcript_id} → {base_id}")
        transcript_id = base_id

    # STEP 2: Attempt sequence fetch with fallback types
    types_to_try = [seq_type] + [t for t in fallback_types if t != seq_type]
    for t in types_to_try:
        logger.info(f"Fetching sequence type '{t}' for transcript {transcript_id}")
        seq_url = f"{ENSEMBL_REST_BASE}/sequence/id/{transcript_id}?type={t}&format=fasta"
        r_seq = requests.get(seq_url, headers={"Content-Type":"text/x-fasta"}, timeout=60)
        if r_seq.ok:
            fasta_str = r_seq.text
            lines = fasta_str.splitlines()
            if lines and lines[0].startswith(">"):
                lines[0] = f">{gene_symbol}|{transcript_id}|{t}"
                fasta_str = "\n".join(lines) + "\n"
            logger.info(f"Successfully fetched sequence for {gene_symbol} (transcript {transcript_id}, type {t})")
            return fasta_str
        else:
            logger.warning(f"Failed fetch for {transcript_id} with type={t}: HTTP {r_seq.status_code}")

    # If all types failed
    raise RuntimeError(f"Sequence fetch failed for transcript {transcript_id} of gene {gene_symbol} with types {types_to_try}")

def build_nuc_fasta(gene_list, output_path):
    with open(output_path, "w") as fw:
        for gene in tqdm(gene_list):
            try:
                fasta = fetch_nucleotide_fasta(gene)
                fw.write(fasta)
            except Exception as e:
                logger.warning(f"Skipping gene {gene} (nucleotide): {e}")
    logger.info(f"Nucleotide FASTA written to {output_path}")

# -------- STEP 2: Encode proteins with ESM2 --------
def encode_with_esm2(fasta_path, model_name="esm2_t33_650M_UR50D", device=None):
    logger.info(f"Loading ESM2 model: {model_name}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    embeddings = {}
    records = list(SeqIO.parse(fasta_path, "fasta"))
    batch_size = 8
    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        batch = [(rec.id, str(rec.seq)) for rec in batch_records]
        labels, seqs, toks = batch_converter(batch)
        toks = toks.to(device)
        with torch.no_grad():
            out = model(toks, repr_layers=[33], return_contacts=False)
        token_reps = out["representations"][33]
        for j, rec in enumerate(batch_records):
            seq = str(rec.seq)
            rep = token_reps[j, 1 : len(seq) + 1]
            seq_embed = rep.mean(dim=0).cpu()
            embeddings[rec.id] = seq_embed
    return embeddings

# -------- STEP 3: Encode nucleotides with Evo2 --------
def encode_with_evo2(fasta_path, model_name="evo2_7b", device=None):
    logger.info(f"Loading Evo2 model: {model_name}")
    # Example from Evo2 docs. :contentReference[oaicite:4]{index=4}
    from evo2 import Evo2
    evo_model = Evo2(model_name)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evo_model = evo_model.to(device)
    embeddings = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq).upper().replace("U", "T")
        input_ids = torch.tensor(evo_model.tokenizer.tokenize(seq), dtype=torch.int64).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = evo_model(input_ids, return_embeddings=True)
        # Choose the layer or embedding you want — example: last_hidden_state mean
        hidden = outputs.last_hidden_state  # shape [1, L, D]
        embed = hidden.mean(dim=1).cpu().squeeze(0)  # [D]
        embeddings[rec.id] = embed
    return embeddings

# -------- MAIN EXECUTION --------
if __name__ == "__main__":
    # Step 1A: protein FASTA
    # build_protein_fasta(GENES, OUTPUT_FASTA_PROTEIN)
    # Step 1B: nucleotide FASTA (you must implement fetch_nucleotide_fasta)
    build_nuc_fasta(GENES, OUTPUT_FASTA_NUC)
    # Step 2: encode proteins
    prot_embeds = encode_with_esm2(OUTPUT_FASTA_PROTEIN)
    torch.save(prot_embeds, EMBED_OUT_PROTEIN)
    logger.info(f"Protein embeddings saved to {EMBED_OUT_PROTEIN}")

    # Step 3: encode nucleotides
    nuc_embeds = encode_with_evo2(OUTPUT_FASTA_NUC)
    torch.save(nuc_embeds, EMBED_OUT_NUC)
    logger.info(f"Nucleotide embeddings saved to {EMBED_OUT_NUC}")

    logger.info("All embeddings complete. You can now feed these into your Virtual Cell Challenge workflow.")
