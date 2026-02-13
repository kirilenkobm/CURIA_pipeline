from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    ref_preprocessed_override: Path | None = None

    @property
    def ultimate_bed(self) -> Path:
        return self.output_dir / "ultimate_isoforms.bed"

    @property
    def ultimate_meta(self) -> Path:
        return self.output_dir / "ultimate_isoforms.tsv"

    @property
    def toga_results_dir(self) -> Path:
        return self.output_dir / "toga_mini_results"

    @property
    def toga_regions(self) -> Path:
        return self.toga_results_dir / "toga_orthologous_regions.tsv"

    @property
    def toga_classification(self) -> Path:
        return self.toga_results_dir / "toga_classification_table.tsv"

    @property
    def rna_toga_regions(self) -> Path:
        return self.output_dir / "rna_orthologous_regions.tsv"

    @property
    def joblists_dir(self) -> Path:
        return self.output_dir / "joblists"

    @property
    def short_joblist(self) -> Path:
        return self.joblists_dir / "short_ncRNA_joblist.txt"

    @property
    def intermediate_sqlite_dir(self) -> Path:
        return self.output_dir / "intermediate_sqlite_dbs"

    @property
    def short_sqlite(self) -> Path:
        return self.intermediate_sqlite_dir / "short_ncRNA_results.sqlite"

    @property
    def mappings_dir(self) -> Path:
        return self.output_dir / "mappings"

    @property
    def short_bed(self) -> Path:
        return self.output_dir / "intermediate_bed_fliles" / "short_rna_annotation_intermediate.bed"

    @property
    def query_regions_clusters(self) -> Path:
        return self.mappings_dir / "query_regions_clusters.json"

    @property
    def chrom_sizes(self) -> Path:
        return self.output_dir / "reference.chrom.sizes.tsv"

    @property
    def merged_query_mapping(self) -> Path:
        return self.mappings_dir / "merged_query_regions_mapping.json"

    @property
    def ultimate_to_query(self) -> Path:
        return self.mappings_dir / "ultimate_to_query.json"

    @property
    def ultimate_to_isoforms(self) -> Path:
        return self.mappings_dir / "ultimate_to_isoforms.json"

    @property
    def long_jobs(self) -> Path:
        return self.joblists_dir / "lnc_rna_preprocessing_jobs.tsv"

    @property
    def preprocessed_reference(self) -> Path:
        if self.ref_preprocessed_override is not None:
            return self.ref_preprocessed_override
        return self.output_dir / "preprocessed_reference_data.json"

    @property
    def query_islands_joblist(self) -> Path:
        return self.joblists_dir / "query_islands_scanner_joblist.txt"

    @property
    def query_islands_sqlite(self) -> Path:
        return self.intermediate_sqlite_dir / "query_islands.sqlite"

    @property
    def query_islands_json(self) -> Path:
        return self.mappings_dir / "query_islands.json"
