"""
B1_download_utrecht.py — Download do dataset Utrecht (Zenodo)
=============================================================
Baixa os arquivos necessários do Zenodo (DOI: 10.5281/zenodo.6906504)
e produz coords.parquet com as coordenadas centróide de cada usina.

Arquivos baixados:
  metadata.csv                          (já presente se B1 rodou antes)
  qcpv.py                               (já presente se B1 rodou antes)
  filtered_pv_power_measurements_ac.csv (~2.9 GB — download resumível)

Saídas:
  data/raw/utrecht/metadata.csv
  data/raw/utrecht/qcpv.py
  data/raw/utrecht/filtered_pv_power_measurements_ac.csv
  data/interim/coords.parquet

Executar:
    python B1_download_utrecht.py

Critério de pronto:
  - Todos os arquivos presentes em data/raw/utrecht/
  - data/interim/coords.parquet salvo com 175 linhas
  - station_id, lat_centroid, lon_centroid, capacity_dc_wp, capacity_dc_kwp
"""

import sys
import time
import hashlib
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg

ZENODO_BASE = "https://zenodo.org/records/6906504/files"

# Arquivos a baixar: (nome_remoto, nome_local, obrigatório)
FILES = [
    ("metadata.csv",                               "metadata.csv",                               True),
    ("qcpv.py",                                    "qcpv.py",                                    True),
    ("filtered_pv_power_measurements_ac.csv",      "filtered_pv_power_measurements_ac.csv",      True),
]

OUT_DIR = cfg.DIRS["raw_utrecht"]


# ── Download com progresso ────────────────────────────────────────────────────

def _download(url: str, dest: Path, label: str) -> None:
    """
    Download com suporte a retomada (-C -) via curl para arquivos grandes.
    Fallback para urlretrieve se curl não estiver disponível.
    """
    if dest.exists():
        size = dest.stat().st_size
        print(f"  {label}: já presente ({size/1e6:.1f} MB) — pulando")
        return

    print(f"  {label}: baixando de {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    import shutil, subprocess
    if shutil.which("curl"):
        # curl com retomada automática e retry
        cmd = [
            "curl", "-L", "--retry", "5", "--retry-delay", "10",
            "-C", "-", "--progress-bar",
            url, "-o", str(dest),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  ERRO: curl retornou código {result.returncode}")
            raise RuntimeError(f"curl falhou para {label}")
        print(f"  {label}: concluído ({dest.stat().st_size/1e6:.1f} MB)")
    else:
        # fallback: urlretrieve (sem retomada)
        start = time.time()
        try:
            def _progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(downloaded / total_size * 100, 100)
                    mb  = downloaded / 1e6
                    tot = total_size / 1e6
                    elapsed = time.time() - start
                    speed   = mb / elapsed if elapsed > 0 else 0
                    print(f"\r    {pct:5.1f}%  {mb:.0f}/{tot:.0f} MB  {speed:.1f} MB/s",
                          end="", flush=True)
            urlretrieve(url, dest, reporthook=_progress)
            print(f"\r    100.0%  {dest.stat().st_size/1e6:.1f} MB — concluído")
        except URLError as e:
            print(f"\n  ERRO: {e}")
            raise


# ── Parsear metadados → coords.parquet ───────────────────────────────────────

def build_coords(meta_path: Path) -> pd.DataFrame:
    """
    Lê metadata.csv e constrói DataFrame com centróides e capacidade.

    Colunas de bounding box: north, south, west, east (graus decimais)
    Centróide: lat = (north + south) / 2, lon = (west + east) / 2
    Capacidade DC: estimated_dc_capacity em Watts
    """
    meta = pd.read_csv(meta_path, sep=";")

    # Remover colunas Unnamed
    meta = meta.loc[:, ~meta.columns.str.startswith("Unnamed")]

    coords = pd.DataFrame({
        "station_id":      meta["ID"],
        "begin_ts":        pd.to_datetime(meta["begin_ts"], utc=True),
        "end_ts":          pd.to_datetime(meta["end_ts"], utc=True),
        "lat_centroid":    (meta["north"].astype(float) + meta["south"].astype(float)) / 2,
        "lon_centroid":    (meta["west"].astype(float)  + meta["east"].astype(float))  / 2,
        "lat_north":       meta["north"].astype(float),
        "lat_south":       meta["south"].astype(float),
        "lon_west":        meta["west"].astype(float),
        "lon_east":        meta["east"].astype(float),
        "capacity_dc_wp":  meta["estimated_dc_capacity"].astype(float),
        "capacity_ac_wp":  meta["estimated_ac_capacity"].astype(float),
        "tilt_deg":        meta["tilt"].astype(float),
        "azimuth_deg":     meta["azimuth"].astype(float),
    })
    coords["capacity_dc_kwp"] = coords["capacity_dc_wp"] / 1000.0
    return coords


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "─" * 60
    print(sep)
    print("B1 — DOWNLOAD UTRECHT (Zenodo 10.5281/zenodo.6906504)")
    print(sep)

    # 1. Download dos arquivos
    for remote_name, local_name, required in FILES:
        url  = f"{ZENODO_BASE}/{remote_name}"
        dest = OUT_DIR / local_name
        try:
            _download(url, dest, local_name)
        except Exception as e:
            if required:
                print(f"\nArquivo obrigatório falhou: {local_name}")
                print("Alternativa manual:")
                print(f"  curl -L '{url}' -o '{dest}'")
                sys.exit(1)

    # 2. Verificar integridade básica
    print(f"\n{sep}")
    print("Verificando arquivos:")
    ok = True
    for _, local_name, required in FILES:
        dest = OUT_DIR / local_name
        if dest.exists():
            size_mb = dest.stat().st_size / 1e6
            print(f"  ✓  {local_name:<50s}  {size_mb:8.1f} MB")
        else:
            icon = "✗" if required else "~"
            print(f"  {icon}  {local_name:<50s}  ausente")
            if required:
                ok = False

    if not ok:
        print("\nB1 REPROVADO: arquivos obrigatórios faltando.")
        sys.exit(1)

    # 3. Construir coords.parquet
    print(f"\n{sep}")
    print("Construindo coords.parquet...")
    meta_path  = OUT_DIR / "metadata.csv"
    coords     = build_coords(meta_path)
    out_coords = cfg.DIRS["interim"] / "coords.parquet"
    coords.to_parquet(out_coords, index=False)

    print(f"  Salvo: {out_coords.relative_to(cfg.ROOT)}")
    print(f"  Linhas: {len(coords)}  (esperado: {cfg.UTRECHT['n_stations']})")
    print(f"  Extensão: lat [{coords['lat_centroid'].min():.4f}, {coords['lat_centroid'].max():.4f}]")
    print(f"             lon [{coords['lon_centroid'].min():.4f}, {coords['lon_centroid'].max():.4f}]")
    print(f"  Cap. DC: {coords['capacity_dc_kwp'].min():.2f} – {coords['capacity_dc_kwp'].max():.2f} kWp")

    assert len(coords) == cfg.UTRECHT["n_stations"], \
        f"Esperado {cfg.UTRECHT['n_stations']} usinas, encontrado {len(coords)}"

    print(f"\n{sep}")
    print("B1 concluído — dados Utrecht presentes e coords.parquet salvo.")


if __name__ == "__main__":
    main()
