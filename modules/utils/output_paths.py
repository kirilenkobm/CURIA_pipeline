from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    ref_preprocessed_override: Path | None = None

    # --- Final output directories ---

    @property
    def query_annotation_dir(self) -> Path:
        return self.output_dir / "query_annotation"

    @property
    def toga_results_dir(self) -> Path:
        return self.output_dir / "toga_results"

    @property
    def mappings_dir(self) -> Path:
        return self.output_dir / "mappings"

    # --- Union transcripts ---

    @property
    def union_bed(self) -> Path:
        return self.output_dir / "reference_union_transcripts.bed"

    @property
    def union_meta(self) -> Path:
        return self.output_dir / "reference_union_transcripts_metadata.tsv"

    # --- TOGA results ---

    @property
    def toga_regions(self) -> Path:
        return self.toga_results_dir / "toga_orthologous_regions.tsv"

    @property
    def toga_classification(self) -> Path:
        return self.toga_results_dir / "original_toga_classification_table.tsv"

    @property
    def rna_toga_regions(self) -> Path:
        return self.toga_results_dir / "rna_orthologous_regions.tsv"

    @property
    def chrom_sizes(self) -> Path:
        return self.toga_results_dir / "reference_chrom_sizes.tsv"

    # --- Query annotation BED/TSV outputs (final) ---

    @property
    def short_bed(self) -> Path:
        return self.query_annotation_dir / "short_ncRNA.bed"

    @property
    def short_tsv(self) -> Path:
        return self.query_annotation_dir / "short_ncRNA_details.tsv"

    @property
    def reference_islands_bed(self) -> Path:
        return self.query_annotation_dir / "reference_islands.bed"

    @property
    def query_islands_bed(self) -> Path:
        return self.query_annotation_dir / "query_islands.bed"

    @property
    def aligned_islands_ref_bed(self) -> Path:
        return self.query_annotation_dir / "lncRNA_islands_reference.bed"

    @property
    def aligned_islands_query_bed(self) -> Path:
        return self.query_annotation_dir / "lncRNA_islands.bed"

    # --- Temporary / internal paths ---

    @property
    def joblists_dir(self) -> Path:
        return self.output_dir / "joblists"

    @property
    def short_joblist(self) -> Path:
        return self.joblists_dir / "short_ncRNA_joblist.txt"

    @property
    def query_islands_joblist(self) -> Path:
        return self.joblists_dir / "query_islands_scanner_joblist.txt"

    @property
    def island_alignment_joblist(self) -> Path:
        return self.joblists_dir / "island_alignment_joblist.txt"

    @property
    def intermediate_sqlite_dir(self) -> Path:
        return self.output_dir / "intermediate_sqlite_dbs"

    @property
    def short_sqlite(self) -> Path:
        return self.intermediate_sqlite_dir / "short_ncRNA_results.sqlite"

    @property
    def reference_islands_sqlite(self) -> Path:
        return self.intermediate_sqlite_dir / "reference_islands.sqlite"

    @property
    def query_islands_sqlite(self) -> Path:
        return self.intermediate_sqlite_dir / "query_islands.sqlite"

    @property
    def island_alignment_sqlite(self) -> Path:
        return self.intermediate_sqlite_dir / "island_alignment_results.sqlite"

    # --- Mappings ---

    @property
    def query_regions_clusters(self) -> Path:
        return self.mappings_dir / "query_regions_clusters.json"

    @property
    def union_to_query(self) -> Path:
        return self.mappings_dir / "union_to_query.json"

    @property
    def union_to_isoforms(self) -> Path:
        return self.mappings_dir / "union_to_isoforms.json"

    @property
    def query_islands_json(self) -> Path:
        return self.mappings_dir / "query_islands.json"

    # --- Top-level results ---

    @property
    def preprocessed_reference(self) -> Path:
        if self.ref_preprocessed_override is not None:
            return self.ref_preprocessed_override
        return self.output_dir / "preprocessed_reference_data.json"

    @property
    def island_alignment_results(self) -> Path:
        return self.output_dir / "island_alignment_results.tsv"
